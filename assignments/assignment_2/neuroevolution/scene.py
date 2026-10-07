"""Builds the simulated scene: the world, the robot and the target.

Scoring and video replays both use this, so a video always shows the scene
that was scored. It also fixes two ARIEL quirks (details in the functions):
bumpy terrain is generated from a fixed seed so it is the same every time,
and the robot starts resting on the ground rather than stuck inside it.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import (
    AmphitheatreTerrainWorld,
    CompoundWorld,
    CraterTerrainWorld,
    RuggedTerrainWorld,
    SimpleFlatWorld,
)
from ariel.simulation.environments.heightmap_functions import (
    amphitheater_heightmap,
    crater_heightmap,
    smooth_edges_heightmap,
)
from ariel.utils.noise_gen import PerlinNoise

from .config import ExperimentConfig

#: John Set bodies, by name. The brief requires the body to come from here.
BODIES: dict[str, Any] = {
    name: function
    for name, function in inspect.getmembers(john_set, inspect.isfunction)
    if function.__module__ == john_set.__name__
}

#: Passive settling per round before the pose is frozen (simulated seconds).
SETTLE_SECONDS = 2.0
#: A frozen pose counts as resting if a passive robot moves less than this.
REST_TOLERANCE = 0.01
MAX_SETTLE_ROUNDS = 10
#: Penetration tolerated at spawn; MuJoCo contacts are soft, so not exactly 0.
PENETRATION_TOLERANCE = 1e-3

#: Worlds the brief excludes, and worlds whose terrain cannot be fixed.
UNSUPPORTED_WORLDS = {
    "tilted": "excluded by the brief: John Set bodies slide off it",
    "olympic": "its terrain noise is unseeded and cannot be replaced via the public API",
}


@dataclass
class Scene:
    """A compiled scene plus the handles the controller and evaluator need."""

    model: mj.MjModel
    data: mj.MjData
    core_body: int            # body owning the robot's free joint
    core_qpos: int            # qpos address of the core's (x, y, z, quaternion)
    hinge_qpos: np.ndarray    # qpos address of each actuated hinge, in actuator order
    spawn_xy: np.ndarray
    target_xy: np.ndarray


# --------------------------------------------------------------------------- #
#  Worlds
# --------------------------------------------------------------------------- #

def _defaults(world_class: type, overrides: dict[str, Any]) -> dict[str, Any]:
    """The world class's own default parameters, with the config's overrides."""
    params = {
        f.name: f.default
        for f in dataclasses.fields(world_class)
        if f.default is not dataclasses.MISSING
    }
    unknown = set(overrides) - set(params)
    if unknown:
        msg = f"unknown parameters for {world_class.__name__}: {sorted(unknown)}"
        raise ValueError(msg)
    return params | overrides


def _noise(p: dict[str, Any], seed: int) -> np.ndarray:
    """ARIEL's `rugged_heightmap`, but seeded."""
    nrow, ncol = p["dims"]
    return PerlinNoise(seed=seed).as_grid(
        ncol, nrow, scale=p["scale_of_noise"], normalize=p["normalize"]
    )


def _compound(p: dict[str, Any], heightmap: np.ndarray, **extra: Any) -> CompoundWorld:
    return CompoundWorld(
        name=p["name"],
        floor_size=p["floor_size"],
        dims=p["dims"],
        floor_heightmap=heightmap,
        checker_floor=p["checker_floor"],
        load_precompiled=False,
        **extra,
    )


def _crater(overrides: dict[str, Any], seed: int) -> CompoundWorld:
    p = _defaults(CraterTerrainWorld, overrides)
    shape = crater_heightmap(
        dims=p["dims"], crater_depth=p["crater_depth"], crater_radius=p["crater_radius"]
    )
    return _compound(
        p,
        shape + _noise(p, seed) * p["height_of_noise"],
        floor_tilt=p["floor_tilt"],
        floor_rot_sequence=p["floor_rot_sequence"],
    )


def _amphitheatre(overrides: dict[str, Any], seed: int) -> CompoundWorld:
    p = _defaults(AmphitheatreTerrainWorld, overrides)
    shape = amphitheater_heightmap(
        dims=p["dims"],
        ring_inner_radius=p["ring_inner_radius"],
        ring_outer_radius=p["ring_outer_radius"],
        cone_height=p["floor_size"][2],
    )
    return _compound(
        p,
        shape + _noise(p, seed) * p["height_of_noise"],
        floor_tilt=p["floor_tilt"],
        floor_rot_sequence=p["floor_rot_sequence"],
    )


def _rugged(overrides: dict[str, Any], seed: int) -> CompoundWorld:
    p = _defaults(RuggedTerrainWorld, overrides)
    edges = smooth_edges_heightmap(p["dims"], edge_width=0.2)
    return _compound(p, _noise(p, seed) * edges)


def _flat(overrides: dict[str, Any], seed: int) -> SimpleFlatWorld:  # noqa: ARG001
    return SimpleFlatWorld(**overrides)


WORLDS = {
    "flat": _flat,
    "rugged": _rugged,
    "crater": _crater,
    "amphitheatre": _amphitheatre,
}


def build_world(cfg: ExperimentConfig):
    """Build the configured world, with terrain that is the same every time.

    ARIEL's rugged, crater and amphitheatre worlds add random Perlin-noise bumps
    with no seed, so every build makes a new terrain and no global seed can stop
    it. The same genome would then score differently each time, and every run
    would get its own world. So those worlds are rebuilt here with ARIEL's own
    recipe but seeded noise (`terrain_seed`), passed in through the public
    `CompoundWorld` API; no ARIEL source is changed. `OlympicArena` cannot be
    fixed this way and is not supported.

    Raises
    ------
    ValueError
        For an unknown or unsupported world.
    """
    if cfg.world in UNSUPPORTED_WORLDS:
        msg = f"world {cfg.world!r} is not supported: {UNSUPPORTED_WORLDS[cfg.world]}"
        raise ValueError(msg)
    if cfg.world not in WORLDS:
        msg = f"unknown world {cfg.world!r}; choose from {sorted(WORLDS)}"
        raise ValueError(msg)
    return WORLDS[cfg.world](dict(cfg.world_params), cfg.terrain_seed)


# --------------------------------------------------------------------------- #
#  Body and scene
# --------------------------------------------------------------------------- #

def build_body(cfg: ExperimentConfig) -> CoreModule:
    """Build the configured John Set body.

    Raises
    ------
    ValueError
        If the name is not a John Set body.
    """
    if cfg.body not in BODIES:
        msg = f"unknown body {cfg.body!r}; the John Set has {sorted(BODIES)}"
        raise ValueError(msg)
    return BODIES[cfg.body]()


def _add_target_marker(world, target_xy: tuple[float, float]) -> None:
    """A thin red pole at the target, visible in replays.

    A site, not a geom: sites have no collision or mass, so the marker cannot
    affect the physics.
    """
    world.spec.worldbody.add_site(
        name="target_marker",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=[target_xy[0], target_xy[1], 1.0],
        size=[0.03, 1.0, 0.0],
        rgba=[0.85, 0.1, 0.1, 0.8],
    )


def _deepest_penetration(model: mj.MjModel, data: mj.MjData) -> float:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return max((-data.contact[i].dist for i in range(data.ncon)), default=0.0)


def _rest_on_terrain(model: mj.MjModel, data: mj.MjData, core_qpos: int) -> None:
    """Make the robot start resting on the terrain instead of stuck inside it.

    ARIEL's spawn correction ignores heightfield terrain, so on the crater the
    robot starts ~4 cm inside the ground and the physics flings it out (a robot
    with all motors off was thrown 1.3 m up). So the robot is lifted clear,
    left to settle with its motors off, and the resting pose is frozen as the
    start of every evaluation. Settling repeats until a passive robot stays put:
    on bumpy ground one round can leave it jammed into the terrain.

    Edits `model.qpos0`, the pose `mj_resetData` restores.

    Raises
    ------
    RuntimeError
        If the robot cannot be freed from the terrain.
    """
    for _ in range(50):
        depth = _deepest_penetration(model, data)
        if depth <= PENETRATION_TOLERANCE:
            break
        model.qpos0[core_qpos + 2] += depth + 0.005
    else:
        raise RuntimeError("could not lift the robot clear of the terrain at spawn")

    steps = round(SETTLE_SECONDS / model.opt.timestep)
    for _ in range(MAX_SETTLE_ROUNDS):
        start = model.qpos0.copy()
        mj.mj_resetData(model, data)      # motors at rest: ctrl is zero
        mj.mj_step(model, data, nstep=steps)
        drift = np.linalg.norm(data.qpos[core_qpos : core_qpos + 2] - start[core_qpos : core_qpos + 2])
        if drift < REST_TOLERANCE:
            model.qpos0[:] = start        # verified: a passive robot stays here
            break
        model.qpos0[:] = data.qpos        # freeze where it stopped, check again
    else:
        msg = f"robot does not come to rest at spawn after {MAX_SETTLE_ROUNDS} rounds"
        raise RuntimeError(msg)

    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)


def build_scene(cfg: ExperimentConfig) -> Scene:
    """Compile world + body + target into a ready-to-reset scene."""
    mj.set_mjcb_control(None)  # never inherit a stale global controller

    world = build_world(cfg)
    body = build_body(cfg)
    world.spawn(body.spec, position=list(cfg.spawn_position),
                correct_collision_with_floor=True)
    _add_target_marker(world, cfg.target_xy)

    model = world.spec.compile()
    data = mj.MjData(model)

    free = [j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        msg = f"expected one free joint (the robot core), found {len(free)}"
        raise ValueError(msg)
    hinge_joints = model.actuator_trnid[:, 0]
    core_qpos = int(model.jnt_qposadr[free[0]])
    _rest_on_terrain(model, data, core_qpos)

    return Scene(
        model=model,
        data=data,
        core_body=int(model.jnt_bodyid[free[0]]),
        core_qpos=core_qpos,
        hinge_qpos=np.asarray(model.jnt_qposadr[hinge_joints], dtype=int),
        spawn_xy=data.qpos[core_qpos : core_qpos + 2].copy(),  # where it settled
        target_xy=np.asarray(cfg.target_xy, dtype=float),
    )
