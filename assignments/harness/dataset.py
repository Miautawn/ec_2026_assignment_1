"""Turns ARIEL databases into one tidy DataFrame with a FROZEN schema.

Everything downstream -- figures, tables, significance tests -- reads only
these columns. That is the contract: as long as an EA writes a standard ARIEL
`Individual` table, this module can summarise it, and nothing else needs to
know which EA ran. Fitness is assumed lower-is-better throughout.

Reconstructing a generation
---------------------------
ARIEL never deletes rows; `EA._commit()` stamps `time_of_birth` once and
refreshes `time_of_death` every generation an individual is still present.

So the population alive during generation `g` is:
    time_of_birth <= g <= time_of_death  AND  requires_eval = 0

Assignment-specific metrics
---------------------------
Two hooks add columns, both mappings from a column name to a function:

`metrics` -- per individual: `f(genotype, tags) -> float`, both parsed from
JSON. Each adds `mean_<name>` over the alive population and `best_<name>` for
that generation's best individual (e.g. module count, mutation step size).

`population_metrics` -- per generation: `f(Generation) -> float`. Each adds one
column `<name>` (e.g. diversity, mutation success rate). Do not start these
names with "mean_": that prefix marks per-individual metrics.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BaseConfig
from .runner import META_FILENAME

#: One row per (variant, seed, generation). Metric columns are appended.
CORE_COLUMNS: tuple[str, ...] = (
    "variant",
    "seed",
    "generation",
    "n_alive",
    "best_fitness",       # best in this generation (min, since lower is better)
    "mean_fitness",
    "worst_fitness",
    "std_fitness",        # spread WITHIN the generation
    "best_so_far",        # cumulative best up to and including this generation
    "evaluations_so_far",
)

#: (parsed genotype, parsed tags) -> number, for one individual.
Metric = Callable[[Any, dict], float]


@dataclass(frozen=True)
class Generation:
    """What a population metric sees for one generation of one run.

    Every frame has the database columns plus `genotype` and `tags`, parsed
    from JSON. `everyone` is indexed by individual id, for looking up parents.
    """

    number: int
    last: int                 # the run's final generation
    alive: pd.DataFrame       # present during this generation
    born: pd.DataFrame        # created (and evaluated) in this generation
    everyone: pd.DataFrame    # the whole run


#: Generation -> number.
PopulationMetric = Callable[[Generation], float]


def _read_individuals(db_path: Path) -> pd.DataFrame:
    """Read the raw `individual` table."""
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql("SELECT * FROM individual", connection)


def _parse(payload: str | None) -> Any:
    """JSON column -> Python object; None if there is nothing to parse."""
    if not payload:
        return None
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return None


def _apply(metric: Metric, genotype: Any, tags: Any) -> float:
    """Evaluate a per-individual metric; NaN if there is no genotype.

    Errors raised by the metric itself are deliberately not caught: a typo in
    a metric should fail loudly, not become a silent column of NaN.
    """
    if genotype is None:
        return float("nan")
    return float(metric(genotype, tags or {}))


def _columns(
    metrics: Mapping[str, Metric], population_metrics: Mapping[str, PopulationMetric]
) -> tuple[str, ...]:
    extra = tuple(f"{kind}_{name}" for name in metrics for kind in ("mean", "best"))
    return CORE_COLUMNS + extra + tuple(population_metrics)


def summarise_run(
    db_path: Path,
    variant: str,
    seed: int,
    metrics: Mapping[str, Metric] | None = None,
    population_metrics: Mapping[str, PopulationMetric] | None = None,
) -> pd.DataFrame:
    """Summarise one database into per-generation rows of the frozen schema."""
    metrics = metrics or {}
    population_metrics = population_metrics or {}
    columns = _columns(metrics, population_metrics)

    raw = _read_individuals(db_path)
    evaluated = raw.loc[raw["requires_eval"] == 0].copy()
    if evaluated.empty:
        return pd.DataFrame(columns=columns)

    evaluated["genotype"] = evaluated["genotype_"].map(_parse)
    evaluated["tags"] = evaluated["tags_"].map(_parse)
    for name, metric in metrics.items():
        evaluated[f"metric:{name}"] = [
            _apply(metric, genotype, tags)
            for genotype, tags in zip(evaluated["genotype"], evaluated["tags"], strict=True)
        ]
    everyone = evaluated.set_index("id", drop=False)

    first_gen = int(evaluated["time_of_birth"].min())
    last_gen = int(evaluated["time_of_death"].max())

    records: list[dict[str, object]] = []
    for generation in range(first_gen, last_gen + 1):
        alive = evaluated.loc[
            (evaluated["time_of_birth"] <= generation)
            & (evaluated["time_of_death"] >= generation)
        ]
        seen = evaluated.loc[evaluated["time_of_birth"] <= generation]

        if alive.empty:
            continue

        fitness = alive["fitness_"].astype(float)
        best_row = alive.loc[fitness.idxmin()]

        record: dict[str, object] = {
            "variant": variant,
            "seed": seed,
            "generation": generation,
            "n_alive": int(len(alive)),
            "best_fitness": float(fitness.min()),
            "mean_fitness": float(fitness.mean()),
            "worst_fitness": float(fitness.max()),
            "std_fitness": float(fitness.std(ddof=0)),
            "best_so_far": float(seen["fitness_"].astype(float).min()),
            "evaluations_so_far": int(len(seen)),
        }
        for name in metrics:
            record[f"mean_{name}"] = float(alive[f"metric:{name}"].mean())
            record[f"best_{name}"] = float(best_row[f"metric:{name}"])
        if population_metrics:
            view = Generation(
                number=generation,
                last=last_gen,
                alive=alive,
                born=evaluated.loc[evaluated["time_of_birth"] == generation],
                everyone=everyone,
            )
            for name, population_metric in population_metrics.items():
                record[name] = float(population_metric(view))
        records.append(record)

    return pd.DataFrame.from_records(records, columns=columns)


def discover_runs(cfg: BaseConfig) -> list[tuple[Path, str, int]]:
    """Find the completed runs of the configured variants and seeds.

    Runs left over from an earlier config (e.g. seeds 5-9 after cutting the
    seed count to 5) are ignored rather than silently mixed in.
    """
    found: list[tuple[Path, str, int]] = []
    for meta_path in sorted(cfg.results_dir.rglob(META_FILENAME)):
        meta = json.loads(meta_path.read_text())
        variant, seed = str(meta["variant"]), int(meta["seed"])
        db_path = meta_path.parent / "database.db"
        if db_path.is_file() and variant in cfg.variants and seed in cfg.seeds:
            found.append((db_path, variant, seed))
    return found


def load(
    cfg: BaseConfig,
    metrics: Mapping[str, Metric] | None = None,
    population_metrics: Mapping[str, PopulationMetric] | None = None,
) -> pd.DataFrame:
    """Load every completed run into one frame.

    Raises
    ------
    FileNotFoundError
        If no completed runs exist yet.
    """
    runs = discover_runs(cfg)
    if not runs:
        msg = f"no completed runs under {cfg.results_dir}. Run the experiment first."
        raise FileNotFoundError(msg)

    frames = [
        summarise_run(db, variant, seed, metrics, population_metrics)
        for db, variant, seed in runs
    ]
    tidy = pd.concat(frames, ignore_index=True)

    # Keep variants in the configured order so every figure's legend matches.
    order = [v for v in cfg.variants if v in set(tidy["variant"])]
    tidy["variant"] = pd.Categorical(tidy["variant"], categories=order, ordered=True)
    return tidy.sort_values(["variant", "seed", "generation"]).reset_index(drop=True)


def metric_names(frame: pd.DataFrame) -> list[str]:
    """Names of the assignment-specific metrics present in a frame."""
    return [
        column.removeprefix("mean_")
        for column in frame.columns
        if column.startswith("mean_") and column not in CORE_COLUMNS
    ]


def final_per_seed(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, seed): the end-of-run outcome.

    This is the unit of statistical analysis -- N equals the number of
    independent runs, never the number of individuals.
    """
    last = tidy.sort_values("generation").groupby(["variant", "seed"], observed=True).tail(1)
    extra = [f"mean_{name}" for name in metric_names(tidy)]
    return last[
        ["variant", "seed", "generation", "best_so_far", "mean_fitness", *extra]
    ].reset_index(drop=True)


def champion_genotypes(cfg: BaseConfig) -> dict[str, dict[str, object]]:
    """Best individual found by each variant across all of its runs.

    Returns a mapping `variant -> {"fitness", "seed", "genotype"}`, where
    `genotype` is the deserialised JSON exactly as the EA stored it.
    """
    best: dict[str, dict[str, object]] = {}
    for db_path, variant, seed in discover_runs(cfg):
        raw = _read_individuals(db_path)
        evaluated = raw.loc[raw["requires_eval"] == 0]
        if evaluated.empty:
            continue
        row = evaluated.loc[evaluated["fitness_"].astype(float).idxmin()]
        fitness = float(row["fitness_"])
        if variant not in best or fitness < float(best[variant]["fitness"]):
            best[variant] = {
                "fitness": fitness,
                "seed": seed,
                "genotype": json.loads(row["genotype_"]),
            }
    return best
