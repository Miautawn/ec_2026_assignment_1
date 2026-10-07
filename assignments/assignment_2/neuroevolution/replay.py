"""Watching stored robots: videos and the live 3D viewer.

Robots are read back from the run databases (nothing extra is stored) and
replayed through the evaluator itself, so what you see is what was scored.
A replay that does not reproduce the stored fitness raises an error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np

from harness.dataset import champion_genotypes

from .config import ExperimentConfig
from .evaluator import EvalResult, Evaluator
from .genotype import Genotype

#: Video frame rate. Must divide `control_hz` (default 50 Hz) evenly, so every
#: frame lands exactly on a network query.
VIDEO_FPS = 25


@dataclass(frozen=True)
class Champion:
    """A stored controller worth replaying."""

    variant: str
    seed: int
    fitness: float
    weights: np.ndarray


def champions(cfg: ExperimentConfig) -> dict[str, Champion]:
    """Each configured variant's best individual over all of its seeds."""
    return {
        variant: Champion(
            variant=variant,
            seed=int(record["seed"]),
            fitness=float(record["fitness"]),
            weights=Genotype.from_json(record["genotype"]).weights,
        )
        for variant, record in champion_genotypes(cfg).items()
    }


def videos_dir(cfg: ExperimentConfig) -> Path:
    return cfg.results_dir.parent / "videos"


def _check(champion: Champion, result: EvalResult) -> EvalResult:
    if result.fitness != champion.fitness:
        msg = (
            f"replay of {champion.variant} (seed {champion.seed}) scored "
            f"{result.fitness!r}, but {champion.fitness!r} was stored: the scene "
            "or config differs from the one the run used"
        )
        raise RuntimeError(msg)
    return result


def _tracking_camera(evaluator: Evaluator) -> mj.MjvCamera:
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = evaluator.scene.core_body
    camera.distance = 2.0
    camera.azimuth = 90.0       # behind the robot, looking along +y at the target
    camera.elevation = -25.0
    return camera


def video(
    cfg: ExperimentConfig,
    champion: Champion,
    path: Path,
    *,
    width: int = 1280,
    height: int = 720,
) -> EvalResult:
    """Render the champion's evaluation to an mp4, camera following the robot."""
    evaluator = Evaluator(cfg)
    model = evaluator.scene.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)    # display
    model.vis.global_.offheight = max(model.vis.global_.offheight, height) # only
    queries_per_frame = round(cfg.control_hz / VIDEO_FPS)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height))
    if not writer.isOpened():
        msg = f"could not open a video writer for {path}"
        raise RuntimeError(msg)

    camera = _tracking_camera(evaluator)
    query = 0
    with mj.Renderer(model, height=height, width=width) as renderer:

        def frame(data: mj.MjData) -> None:
            nonlocal query
            if query % queries_per_frame == 0:
                renderer.update_scene(data, camera=camera)
                writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            query += 1

        try:
            result = evaluator.evaluate(champion.weights, observer=frame)
        finally:
            writer.release()
    print(f"  wrote {path}")
    return _check(champion, result)


class _ViewerClosed(Exception):
    pass


def view(cfg: ExperimentConfig, champion: Champion) -> None:
    """Replay the champion in MuJoCo's interactive viewer, in real time, on a
    loop, until the window is closed. (On macOS, run with `mjpython`.)
    """
    from mujoco import viewer

    evaluator = Evaluator(cfg)
    seconds_per_query = 1.0 / cfg.control_hz
    with viewer.launch_passive(evaluator.scene.model, evaluator.scene.data) as window:
        window.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        window.cam.trackbodyid = evaluator.scene.core_body
        window.cam.distance = 2.0
        window.cam.azimuth = 90.0
        window.cam.elevation = -25.0

        def show(data: mj.MjData) -> None:  # noqa: ARG001
            if not window.is_running():
                raise _ViewerClosed
            window.sync()
            time.sleep(seconds_per_query)

        while window.is_running():
            try:
                _check(champion, evaluator.evaluate(champion.weights, observer=show))
            except _ViewerClosed:
                return
