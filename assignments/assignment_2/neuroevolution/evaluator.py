"""Scores a genome: runs one simulation and measures where the robot ended up.

The scene is built once and reset before every evaluation, which gives the same
result as rebuilding it, only cheaper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from .config import ExperimentConfig
from .controller import NeuralController, genome_length
from .scene import Scene, build_scene

#: Lower is better: the fitness is a distance to the target.
IS_MAXIMISATION = False

#: MuJoCo's instability warnings. On a blow-up MuJoCo usually does NOT leave
#: NaN behind: it warns, and keeps the state finite (e.g. by dropping bad
#: controls or resetting) -- so a NaN check alone would miss most failures.
#: `mj_resetData` clears these counters, so each evaluation starts at zero.
_INSTABILITY = (
    mj.mjtWarning.mjWARN_BADQPOS,
    mj.mjtWarning.mjWARN_BADQVEL,
    mj.mjtWarning.mjWARN_BADQACC,
    mj.mjtWarning.mjWARN_BADCTRL,
)


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one evaluation.

    Attributes
    ----------
    fitness
        Planar distance from the core's final position to the target, in
        metres. For a failed evaluation, the start distance ("did not move").
    final_xy
        The core's final (x, y).
    distance_from_spawn
        Planar distance travelled from spawn -- e.g. how far up a crater slope
        the robot got, whatever its direction.
    failed
        True when MuJoCo flagged the simulation as unstable (or the state went
        non-finite). The robot's position is then meaningless, so it is
        scored as if it never moved.
    trajectory
        Core (x, y, z) at every network query, if requested; else None.
    """

    fitness: float
    final_xy: tuple[float, float]
    distance_from_spawn: float
    failed: bool
    trajectory: np.ndarray | None = None

    def as_tags(self) -> dict[str, object]:
        """JSON-friendly summary to store on an Individual (no trajectory)."""
        return {
            "final_x": self.final_xy[0],
            "final_y": self.final_xy[1],
            "distance_from_spawn": self.distance_from_spawn,
            "failed": self.failed,
        }


class Evaluator:
    """Scores genomes on one fixed, reusable scene."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.scene: Scene = build_scene(cfg)
        model = self.scene.model

        physics_hz = 1.0 / model.opt.timestep
        ratio = physics_hz / cfg.control_hz
        if cfg.control_hz <= 0 or ratio < 1 or not np.isclose(ratio, round(ratio)):
            msg = (
                f"control_hz={cfg.control_hz} must divide the physics rate "
                f"({physics_hz:.0f} Hz) evenly"
            )
            raise ValueError(msg)
        self.steps_per_query = round(ratio)
        self.n_steps = round(cfg.sim_duration / model.opt.timestep)

    @property
    def n_hinges(self) -> int:
        return len(self.scene.hinge_qpos)

    @property
    def genome_length(self) -> int:
        return genome_length(self.n_hinges, self.cfg.hidden_size)

    def reset(self) -> None:
        """Return the scene to its exact starting state."""
        mj.mj_resetData(self.scene.model, self.scene.data)
        mj.mj_forward(self.scene.model, self.scene.data)

    def unstable(self) -> bool:
        """Whether MuJoCo has flagged this simulation as numerically unstable."""
        data = self.scene.data
        return (
            any(data.warning[int(w)].number for w in _INSTABILITY)
            or not np.isfinite(data.qpos).all()
        )

    def core_xy(self) -> np.ndarray:
        q = self.scene.core_qpos
        return self.scene.data.qpos[q : q + 2].copy()

    def evaluate(
        self,
        genome: Sequence[float],
        *,
        record_trajectory: bool = False,
        observer: Callable[[mj.MjData], None] | None = None,
    ) -> EvalResult:
        """Run one full simulation of `genome` and score it.

        The controller is called explicitly, never through MuJoCo's global callback,
        so evaluations are safe to run in parallel processes.

        `observer`, if given, is called at every network query with the live
        simulation state: this is how videos and the viewer watch the exact loop
        that is scored. It must only read the state, never change it.
        """
        model, data = self.scene.model, self.scene.data
        self.reset()
        controller = NeuralController(
            genome, self.scene, self.cfg.hidden_size, self.cfg.clock_hz
        )
        spawn = self.core_xy()
        trajectory: list[np.ndarray] = []
        failed = False

        # Between queries the commands are held, so step the physics in one
        # chunk per query (identical to stepping one at a time, ~10x fewer
        # Python iterations).
        for start in range(0, self.n_steps, self.steps_per_query):
            if self.unstable():
                failed = True
                break
            data.ctrl[:] = controller(data)
            if record_trajectory:
                q = self.scene.core_qpos
                trajectory.append(data.qpos[q : q + 3].copy())
            if observer is not None:
                observer(data)
            mj.mj_step(model, data, nstep=min(self.steps_per_query, self.n_steps - start))

        final = self.core_xy()
        failed = failed or self.unstable()
        if failed:
            final = spawn
        fitness = float(np.linalg.norm(self.scene.target_xy - final))

        return EvalResult(
            fitness=fitness,
            final_xy=(float(final[0]), float(final[1])),
            distance_from_spawn=float(np.linalg.norm(final - spawn)),
            failed=failed,
            trajectory=np.asarray(trajectory) if record_trajectory else None,
        )
