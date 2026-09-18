"""Turns ARIEL databases into one tidy DataFrame with a FROZEN schema.

Everything downstream -- figures, tables, significance tests -- reads only the
columns documented in `PER_GENERATION_COLUMNS`. That is the contract: as long
as an EA writes a standard ARIEL `Individual` table, this module can summarise
it, and nothing else needs to know which EA ran.

Reconstructing a generation
---------------------------
ARIEL never deletes rows; `EA._commit()` stamps `time_of_birth` once and
refreshes `time_of_death` every generation an individual is still present.

So the population alive during generation `g` is:
    time_of_birth <= g <= time_of_death  AND  requires_eval = 0
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .runner import META_FILENAME

#: The frozen schema. One row per (variant, seed, generation).
PER_GENERATION_COLUMNS: tuple[str, ...] = (
    "variant",
    "seed",
    "generation",
    "n_alive",
    "best_fitness",       # best in this generation (min, since lower is better)
    "mean_fitness",
    "worst_fitness",
    "std_fitness",        # spread WITHIN the generation (not the fitness's own std term)
    "best_so_far",        # cumulative best up to and including this generation
    "evaluations_so_far",
    "mean_size",          # mean module count -- bloat
    "best_size",          # module count of this generation's best body
)


def _read_individuals(db_path: Path) -> pd.DataFrame:
    """Read the raw `individual` table."""
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql("SELECT * FROM individual", connection)


def _module_count(genotype: str | None) -> float:
    """Number of modules in a serialised tree genotype.
    """
    if not genotype:
        return float("nan")
    try:
        payload = json.loads(genotype)
        return float(len(payload["nodes"]))
    except (TypeError, json.JSONDecodeError):
        return float("nan")

def summarise_run(db_path: Path, variant: str, seed: int) -> pd.DataFrame:
    """Summarise one database into per-generation rows of the frozen schema."""
    raw = _read_individuals(db_path)
    evaluated = raw.loc[raw["requires_eval"] == 0].copy()
    if evaluated.empty:
        return pd.DataFrame(columns=PER_GENERATION_COLUMNS)

    evaluated["size"] = evaluated["genotype_"].map(_module_count)

    first_gen = int(evaluated["time_of_birth"].min())
    last_gen = int(evaluated["time_of_death"].max())

    records: list[dict[str, object]] = []
    for generation in range(first_gen, last_gen + 1):
        alive = evaluated.loc[
            (evaluated["time_of_birth"] <= generation)
            & (evaluated["time_of_death"] >= generation)
        ]
        seen = evaluated.loc[evaluated["time_of_birth"] <= generation]
        born = evaluated.loc[evaluated["time_of_birth"] == generation]

        if alive.empty:
            continue

        fitness = alive["fitness_"].astype(float)
        best_row = alive.loc[fitness.idxmin()]

        records.append(
            {
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
                "mean_size": float(alive["size"].mean()),
                "best_size": float(best_row["size"]),
            }
        )

    return pd.DataFrame.from_records(records, columns=PER_GENERATION_COLUMNS)


def discover_runs(cfg: ExperimentConfig) -> list[tuple[Path, str, int]]:
    """Find every completed run under the results directory.
    """
    found: list[tuple[Path, str, int]] = []
    for meta_path in sorted(cfg.results_dir.rglob(META_FILENAME)):
        meta = json.loads(meta_path.read_text())
        db_path = meta_path.parent / "database.db"
        if db_path.is_file():
            found.append((db_path, str(meta["variant"]), int(meta["seed"])))
    return found


def load(cfg: ExperimentConfig) -> pd.DataFrame:
    """Load every completed run into one frame.

    Raises
    ------
    FileNotFoundError
        If no completed runs exist yet.
    """
    runs = discover_runs(cfg)
    if not runs:
        msg = (
            f"no completed runs under {cfg.results_dir}. "
            "Run `uv run assignments/assignment_1/run_experiment.py` first."
        )
        raise FileNotFoundError(msg)

    frames = [summarise_run(db, variant, seed) for db, variant, seed in runs]
    tidy = pd.concat(frames, ignore_index=True)

    # Keep variants in the configured order so every figure's legend matches.
    order = [v for v in cfg.variants if v in set(tidy["variant"])]
    tidy["variant"] = pd.Categorical(tidy["variant"], categories=order, ordered=True)
    return tidy.sort_values(["variant", "seed", "generation"]).reset_index(drop=True)


def final_per_seed(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, seed): the end-of-run outcome.

    This is the unit of statistical analysis -- N equals the number of
    independent runs, never the number of individuals.
    """
    last = tidy.sort_values("generation").groupby(["variant", "seed"], observed=True).tail(1)
    return last[
        ["variant", "seed", "generation", "best_so_far", "mean_fitness", "mean_size"]
    ].reset_index(drop=True)


def champion_genotypes(cfg: ExperimentConfig) -> dict[str, dict[str, object]]:
    """Best individual found by each variant across all of its runs.

    Returns a mapping `variant -> {"fitness", "seed", "genotype"}`, where
    `genotype` is the deserialised JSON exactly as the EA stored it. Used for
    the per-target breakdown figure and for eyeballing the final bodies.
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
