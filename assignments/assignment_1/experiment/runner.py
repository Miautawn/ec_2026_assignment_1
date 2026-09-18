"""Executes the (variant x seed) grid, one database per run.

Each run gets its own directory holding `database.db` plus a `meta.json`
recording the variant, seed, wall time, realised evaluation count and the full
config.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path

from . import variants
from .config import ExperimentConfig

META_FILENAME = "meta.json"


def count_evaluations(db_path: Path) -> int:
    """Realised number of fitness evaluations, read back from the database.

    Every individual is evaluated exactly once, so the row count *is* the
    evaluation count. Reading it back rather than trusting the configured
    budget is what lets the report claim the budgets were genuinely equal.
    """
    with sqlite3.connect(db_path) as connection:
        (rows,) = connection.execute(
            "SELECT COUNT(*) FROM individual WHERE requires_eval = 0"
        ).fetchone()
    return int(rows)


def execute(variant: str, seed: int, cfg: ExperimentConfig) -> dict[str, object]:
    """Run one complete EA loop from scratch and write its metadata."""
    run_dir = cfg.run_dir(variant, seed)
    db_path = cfg.db_path(variant, seed)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    started = time.perf_counter()
    variants.build(variant, seed, db_path, cfg).run()
    elapsed = time.perf_counter() - started

    meta = {
        "variant": variant,
        "seed": seed,
        "wall_seconds": round(elapsed, 2),
        "evaluations": count_evaluations(db_path),
        "budget": cfg.evaluation_budget,
        "config": cfg.as_dict(),
    }
    (run_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))
    return meta


def run_all(cfg: ExperimentConfig) -> list[dict[str, object]]:
    """Run the whole grid."""
    grid = [(variant, seed) for variant in cfg.variants for seed in cfg.seeds]
    metas: list[dict[str, object]] = []

    for index, (variant, seed) in enumerate(grid, start=1):
        tag = f"[{index:>3}/{len(grid)}] {variant} seed={seed:02d}"
        print(f"{tag}  running...", flush=True)
        meta = execute(variant, seed, cfg)
        print(f"{tag}  done in {meta['wall_seconds']}s, "
              f"{meta['evaluations']} evaluations")
        metas.append(meta)

    counts = set(int(meta["evaluations"]) for meta in metas)
    if len(counts) > 1:
        print(f"\n*** WARNING: unequal evaluation counts {sorted(counts)}. "
              "The comparison is not budget-matched. ***\n")
    return metas
