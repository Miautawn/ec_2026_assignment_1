"""Scene, controller and evaluator: determinism, fixed terrain, wiring."""

import sys
from dataclasses import replace
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroevolution.config import ExperimentConfig
from neuroevolution.controller import NeuralController, genome_length
from neuroevolution.evaluator import Evaluator
from neuroevolution.scene import build_scene

from ariel.simulation.environments import AmphitheatreTerrainWorld, CraterTerrainWorld


@pytest.fixture
def cfg():
    return ExperimentConfig(sim_duration=1.0)


def genome(evaluator, seed=0, scale=0.5):
    return np.random.default_rng(seed).normal(scale=scale, size=evaluator.genome_length)


def heightfield(cfg):
    return build_scene(cfg).model.hfield_data.copy()


def test_gecko_genome_has_150_genes(cfg):
    evaluator = Evaluator(cfg)
    assert evaluator.n_hinges == 6
    assert evaluator.genome_length == genome_length(6, 8) == 150


def test_same_genome_same_result_on_reused_and_fresh_scene(cfg):
    evaluator = Evaluator(cfg)
    g = genome(evaluator)
    first = evaluator.evaluate(g)
    assert evaluator.evaluate(g) == first
    assert Evaluator(cfg).evaluate(g) == first
    assert not first.failed


@pytest.mark.parametrize("world", ["crater", "rugged", "amphitheatre"])
def test_terrain_is_fixed_by_the_terrain_seed(world, cfg):
    cfg = replace(cfg, world=world)
    assert np.array_equal(heightfield(cfg), heightfield(cfg))
    assert not np.array_equal(heightfield(cfg), heightfield(replace(cfg, terrain_seed=1)))


@pytest.mark.parametrize(
    "world,ariel_class", [("crater", CraterTerrainWorld), ("amphitheatre", AmphitheatreTerrainWorld)]
)
def test_terrain_recipe_matches_ariel_apart_from_the_noise(world, ariel_class, cfg):
    """With the noise switched off, our rebuild must equal ARIEL's own world."""
    ours = heightfield(replace(cfg, world=world, world_params={"height_of_noise": 0.0}))
    mj.set_mjcb_control(None)
    theirs = ariel_class(height_of_noise=0.0).spec.compile().hfield_data
    np.testing.assert_allclose(ours, theirs)


def test_inputs_at_spawn(cfg):
    # Flat ground: on the crater the robot settles slightly rotated (by design).
    evaluator = Evaluator(replace(cfg, world="flat"))
    evaluator.reset()
    inputs = NeuralController(genome(evaluator), evaluator.scene, 8, 1.0).inputs(
        evaluator.scene.data
    )
    assert inputs.shape == (11,)
    clock, bearing, distance = inputs[6:8], inputs[8:10], inputs[10]
    np.testing.assert_allclose(clock, [0.0, 1.0], atol=1e-9)        # t = 0
    np.testing.assert_allclose(bearing, [1.0, 0.0], atol=1e-3)      # straight ahead
    assert distance == pytest.approx(1.0, abs=1e-3)


def test_target_to_the_left_has_positive_bearing(cfg):
    evaluator = Evaluator(replace(cfg, world="flat"))
    evaluator.reset()
    evaluator.scene.target_xy = np.array([-3.0, 0.0])  # robot faces +y; -x is its left
    controller = NeuralController(genome(evaluator), evaluator.scene, 8, 1.0)
    cos_b, sin_b = controller.inputs(evaluator.scene.data)[8:10]
    assert sin_b == pytest.approx(1.0, abs=1e-2)
    assert cos_b == pytest.approx(0.0, abs=1e-2)


def test_network_is_queried_at_control_hz(cfg, monkeypatch):
    calls = []
    original = NeuralController.__call__

    def counted(self, data):
        calls.append(data.time)
        return original(self, data)

    monkeypatch.setattr(NeuralController, "__call__", counted)
    result = Evaluator(cfg).evaluate(np.zeros(150), record_trajectory=True)
    assert len(calls) == 50                       # 1 s at 50 Hz
    assert np.diff(calls) == pytest.approx(0.02)  # every 10 physics steps
    assert result.trajectory.shape == (50, 3)


def test_unstable_simulation_is_flagged_and_scored_as_not_moving(cfg, monkeypatch):
    monkeypatch.setattr(NeuralController, "__call__", lambda self, data: np.full(6, np.nan))
    evaluator = Evaluator(cfg)
    result = evaluator.evaluate(np.zeros(150))
    assert result.failed
    assert result.fitness == pytest.approx(np.linalg.norm(evaluator.scene.target_xy - evaluator.scene.spawn_xy), abs=1e-2)
    assert result.distance_from_spawn == 0.0


@pytest.mark.parametrize(
    "change,message",
    [
        ({"world": "tilted"}, "excluded by the brief"),
        ({"world": "olympic"}, "unseeded"),
        ({"world": "moon"}, "unknown world"),
        ({"body": "dragon"}, "unknown body"),
        ({"world_params": {"lava": 1}}, "unknown parameters"),
        ({"control_hz": 300.0}, "divide"),
    ],
)
def test_invalid_configuration_is_rejected(change, message, cfg):
    with pytest.raises(ValueError, match=message):
        Evaluator(replace(cfg, **change))


def test_wrong_genome_length_is_rejected(cfg):
    with pytest.raises(ValueError, match="genes"):
        Evaluator(cfg).evaluate(np.zeros(149))


@pytest.mark.parametrize("world", ["crater", "rugged", "amphitheatre", "flat"])
def test_a_passive_robot_rests_instead_of_being_launched(world, cfg):
    """Regression: the robot used to spawn inside the terrain and be ejected.

    With every motor at rest it was thrown 1.3 m up and 2 m sideways on the
    crater. A passive robot must now stay (practically) where it starts; a few
    millimetres of slow, friction-limited creep on a slope is real physics.
    """
    evaluator = Evaluator(replace(cfg, world=world, sim_duration=5.0))
    data = evaluator.scene.data
    evaluator.reset()
    z0 = float(data.qpos[evaluator.scene.core_qpos + 2])
    result = evaluator.evaluate(np.zeros(evaluator.genome_length), record_trajectory=True)
    assert result.distance_from_spawn < 0.05            # launches moved metres
    assert (result.trajectory[:, 2] - z0).max() < 0.02  # launches rose >1 m
