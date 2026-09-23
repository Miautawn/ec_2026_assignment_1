"""Smoke checks for the official Assignment 2 framework update."""

import importlib.util
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from ariel.ec import generators, set_seed


def test_shared_rng_reseeding():
    rng = generators._rng
    state = rng.bit_generator.state
    previous_seed = generators.SEED
    try:
        set_seed(13)
        first = rng.normal(size=10)
        set_seed(14)
        assert not np.array_equal(first, rng.normal(size=10))
        set_seed(13)
        np.testing.assert_array_equal(first, rng.normal(size=10))
        assert generators._rng is rng
    finally:
        set_seed(previous_seed)
        rng.bit_generator.state = state


def test_headless_demo_replays_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[1] / "A2_template_2026.py"
    spec = importlib.util.spec_from_file_location("a2_official_demo", path)
    module = importlib.util.module_from_spec(spec)
    state = generators._rng.bit_generator.state
    previous_seed = generators.SEED
    try:
        spec.loader.exec_module(module)
        module.SIM_DURATION = 0.2
        scores = []
        for _ in range(2):
            module.RNG = np.random.default_rng(42)
            set_seed(42)
            scores.append(module.run_experiment("simple"))
            assert mj.get_mjcb_control() is None
        assert np.isfinite(scores).all()
        assert min(scores) >= 0
        assert scores[0] == pytest.approx(scores[1], abs=1e-12, rel=0)
    finally:
        mj.set_mjcb_control(None)
        set_seed(previous_seed)
        generators._rng.bit_generator.state = state
