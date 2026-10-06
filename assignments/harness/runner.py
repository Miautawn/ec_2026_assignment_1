"""Executes the (variant x seed) grid, one database per run, in parallel.

Each run gets its own directory holding `database.db` plus a `meta.json`
recording the variant, seed, wall time, realised evaluation count and the full
config.

Runs are independent, so they execute in separate processes, `cfg.workers` at
a time. Processes use the "spawn" start method on every OS, so Linux, macOS and
Windows behave the same; the price is a few seconds of start-up per worker.
Results do not depend on the number of workers.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sqlite3
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import BaseConfig
from .variants import Variant

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


def execute(
    variant: str, seed: int, cfg: BaseConfig, registry: Mapping[str, Variant]
) -> dict[str, object]:
    """Run one complete EA loop from scratch and write its metadata."""
    run_dir = cfg.run_dir(variant, seed)
    db_path = cfg.db_path(variant, seed)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    started = time.perf_counter()
    registry[variant].build(seed, db_path, cfg).run()
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


def resolve_workers(cfg: BaseConfig, n_runs: int) -> int:
    """Worker processes to use: the configured count, or every core."""
    workers = cfg.workers if cfg.workers is not None else (os.cpu_count() or 1)
    if workers < 1:
        msg = f"workers must be at least 1, got {workers}"
        raise ValueError(msg)
    return max(1, min(workers, n_runs))


def _report(index: int, total: int, meta: dict[str, object]) -> None:
    print(f"[{index:>3}/{total}] {meta['variant']} seed={meta['seed']:02d}  "
          f"done in {meta['wall_seconds']}s, {meta['evaluations']} evaluations",
          flush=True)


def run_all(cfg: BaseConfig, registry: Mapping[str, Variant]) -> list[dict[str, object]]:
    """Run the whole grid, `cfg.workers` runs at a time."""
    grid = [(variant, seed) for variant in cfg.variants for seed in cfg.seeds]
    workers = resolve_workers(cfg, len(grid))

    print(f"  variants  : {list(cfg.variants)}")
    print(f"  seeds     : {list(cfg.seeds)}  ({len(cfg.seeds)} independent runs each)")
    print(f"  budget    : {cfg.evaluation_budget} evaluations per run "
          f"(pop {cfg.population_size} + {cfg.generations} gen x "
          f"{cfg.offspring_per_generation} offspring)")
    print(f"  workers   : {workers} parallel process(es) for {len(grid)} runs")
    print(f"  results   : {cfg.results_dir}\n", flush=True)

    metas: list[dict[str, object]] = []
    if workers == 1:
        for index, (variant, seed) in enumerate(grid, start=1):
            metas.append(execute(variant, seed, cfg, registry))
            _report(index, len(grid), metas[-1])
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(execute, variant, seed, cfg, registry): (variant, seed)
                for variant, seed in grid
            }
            try:
                for index, future in enumerate(as_completed(futures), start=1):
                    variant, seed = futures[future]
                    try:
                        metas.append(future.result())
                    except Exception as error:
                        msg = f"run {variant} seed={seed} failed: {error}"
                        raise RuntimeError(msg) from error
                    _report(index, len(grid), metas[-1])
            except BaseException:
                pool.shutdown(cancel_futures=True)  # don't wait for the rest
                raise

    order = {variant: i for i, variant in enumerate(cfg.variants)}
    metas.sort(key=lambda meta: (order[meta["variant"]], meta["seed"]))

    counts = set(int(meta["evaluations"]) for meta in metas)
    if len(counts) > 1:
        print(f"\n*** WARNING: unequal evaluation counts {sorted(counts)}. "
              "The comparison is not budget-matched. ***\n")
    return metas
