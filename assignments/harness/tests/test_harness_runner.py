"""Worker-count resolution and which runs the dataset loads."""

import json
import os
from dataclasses import replace

import pytest

from harness.config import BaseConfig
from harness.dataset import discover_runs
from harness.runner import META_FILENAME, resolve_workers


def test_worker_count():
    cfg = BaseConfig()
    assert resolve_workers(cfg, n_runs=1000) == (os.cpu_count() or 1)  # None: every core
    assert resolve_workers(replace(cfg, workers=4), n_runs=1000) == 4
    assert resolve_workers(replace(cfg, workers=99), n_runs=3) == 3       # no idle workers
    with pytest.raises(ValueError, match="at least 1"):
        resolve_workers(replace(cfg, workers=0), n_runs=3)


def test_only_the_configured_grid_is_loaded(tmp_path):
    """Leftovers from an earlier config must not be mixed into the analysis."""
    cfg = BaseConfig(variants=("a",), seeds=(0, 1), results_dir=tmp_path)
    for variant, seed in [("a", 0), ("a", 1), ("a", 7), ("old", 0)]:
        run_dir = cfg.run_dir(variant, seed)
        run_dir.mkdir(parents=True)
        (run_dir / "database.db").touch()
        (run_dir / META_FILENAME).write_text(json.dumps({"variant": variant, "seed": seed}))
    assert sorted((v, s) for _, v, s in discover_runs(cfg)) == [("a", 0), ("a", 1)]
