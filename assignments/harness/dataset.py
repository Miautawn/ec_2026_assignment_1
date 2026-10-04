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
An assignment can pass `metrics`: a mapping from a name to a function of one
individual's (parsed JSON) genotype. Each adds two columns, `mean_<name>` over
the alive population and `best_<name>` for that generation's best individual
-- e.g. module count for tree genomes, mutation step size for self-adaptation.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
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

#: Parsed genotype -> number.
Metric = Callable[[Any], float]


def _read_individuals(db_path: Path) -> pd.DataFrame:
    """Read the raw `individual` table."""
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql("SELECT * FROM individual", connection)


def _apply(metric: Metric, genotype: str | None) -> float:
    """Evaluate a metric on a stored genotype; NaN if there is none to parse.

    Errors raised by the metric itself are deliberately not caught: a typo in
    a metric should fail loudly, not become a silent column of NaN.
    """
    if not genotype:
        return float("nan")
    try:
        payload = json.loads(genotype)
    except (TypeError, ValueError):
        return float("nan")
    return float(metric(payload))


def _columns(metrics: Mapping[str, Metric]) -> tuple[str, ...]:
    extra = tuple(f"{kind}_{name}" for name in metrics for kind in ("mean", "best"))
    return CORE_COLUMNS + extra


def summarise_run(
    db_path: Path,
    variant: str,
    seed: int,
    metrics: Mapping[str, Metric] | None = None,
) -> pd.DataFrame:
    """Summarise one database into per-generation rows of the frozen schema."""
    metrics = metrics or {}
    columns = _columns(metrics)

    raw = _read_individuals(db_path)
    evaluated = raw.loc[raw["requires_eval"] == 0].copy()
    if evaluated.empty:
        return pd.DataFrame(columns=columns)

    for name, metric in metrics.items():
        evaluated[f"metric:{name}"] = evaluated["genotype_"].map(
            lambda genotype, metric=metric: _apply(metric, genotype)
        )

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
        records.append(record)

    return pd.DataFrame.from_records(records, columns=columns)


def discover_runs(cfg: BaseConfig) -> list[tuple[Path, str, int]]:
    """Find every completed run under the results directory."""
    found: list[tuple[Path, str, int]] = []
    for meta_path in sorted(cfg.results_dir.rglob(META_FILENAME)):
        meta = json.loads(meta_path.read_text())
        db_path = meta_path.parent / "database.db"
        if db_path.is_file():
            found.append((db_path, str(meta["variant"]), int(meta["seed"])))
    return found


def load(cfg: BaseConfig, metrics: Mapping[str, Metric] | None = None) -> pd.DataFrame:
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

    frames = [summarise_run(db, variant, seed, metrics) for db, variant, seed in runs]
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
