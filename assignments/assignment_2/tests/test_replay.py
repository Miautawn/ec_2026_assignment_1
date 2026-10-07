"""Replays: they show exactly what was scored."""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroevolution import figures, replay
from neuroevolution.config import ExperimentConfig
from neuroevolution.evaluator import Evaluator

from harness import runner


@pytest.fixture
def cfg(tmp_path):
    return ExperimentConfig(
        variants=("random_search", "toy"), seeds=(0,),
        population_size=4, generations=2, offspring_per_generation=4, sim_duration=0.5,
        results_dir=tmp_path / "results", figures_dir=tmp_path / "figures",
    )


@pytest.fixture
def best(cfg):
    from test_ea import REGISTRY   # random search + the test-only ToyEA

    for variant in cfg.variants:
        runner.execute(variant, 0, cfg, REGISTRY)
    return replay.champions(cfg)


def test_watching_does_not_change_the_simulation(cfg):
    evaluator = Evaluator(cfg)
    genome = np.random.default_rng(0).normal(0, 0.5, evaluator.genome_length)
    seen = []
    watched = evaluator.evaluate(genome, observer=lambda data: seen.append(data.time))
    assert watched == evaluator.evaluate(genome)
    assert len(seen) == 25                      # every query of a 0.5 s run at 50 Hz


def test_champions_come_from_the_database(cfg, best):
    assert set(best) == set(cfg.variants)
    for champion in best.values():
        assert champion.weights.shape == (150,)


def test_video_is_the_scored_simulation(cfg, best, tmp_path):
    champion = best["toy"]
    path = tmp_path / "videos" / "toy.mp4"
    result = replay.video(cfg, champion, path, width=320, height=240)
    assert result.fitness == champion.fitness   # bit-for-bit the stored score
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frames == 13                         # 25 queries, a frame every 2nd (25 fps)


def test_a_replay_that_does_not_reproduce_the_score_is_rejected(cfg, best, tmp_path):
    wrong = replace(best["toy"], fitness=best["toy"].fitness + 1.0)
    with pytest.raises(RuntimeError, match="was stored"):
        replay.video(cfg, wrong, tmp_path / "wrong.mp4", width=320, height=240)


def test_trajectory_figure(cfg, best, tmp_path):
    from test_ea import REGISTRY

    path = figures.trajectories(cfg, best, REGISTRY, tmp_path / "trajectories.png")
    assert path.is_file() and path.stat().st_size > 0
