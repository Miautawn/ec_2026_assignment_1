"""The robot's brain: a small neural network whose weights are the genome.

It reads the robot's state (joint angles, a clock, where the target is) and
outputs a target angle for each motor. The weights are evolved, never trained.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco as mj
import numpy as np

from .scene import Scene

N_TASK_INPUTS = 5  # clock (2) + target bearing (2) + distance (1)
HINGE_RANGE = np.pi / 2


def n_inputs(n_hinges: int) -> int:
    return n_hinges + N_TASK_INPUTS


def genome_length(n_hinges: int, hidden: int) -> int:
    """Number of genes for a body with `n_hinges` actuated hinges."""
    n_in, n_out = n_inputs(n_hinges), n_hinges
    return n_in * hidden + hidden + hidden * n_out + n_out


def decode(
    genome: Sequence[float], n_hinges: int, hidden: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice a flat genome into the network's weights (W1, b1, W2, b2).

    The genome is read in this order: W1 (inputs x hidden, row by row), b1
    (hidden), W2 (hidden x outputs, row by row), b2 (outputs).

    Raises
    ------
    ValueError
        If the genome has the wrong length for this body and hidden size.
    """
    genes = np.asarray(genome, dtype=float)
    expected = genome_length(n_hinges, hidden)
    if genes.shape != (expected,):
        msg = f"genome has {genes.size} genes; this body and network need {expected}"
        raise ValueError(msg)

    n_in, n_out = n_inputs(n_hinges), n_hinges
    sizes = (n_in * hidden, hidden, hidden * n_out, n_out)
    w1, b1, w2, b2 = np.split(genes, np.cumsum(sizes)[:-1])
    return w1.reshape(n_in, hidden), b1, w2.reshape(hidden, n_out), b2


class NeuralController:
    """Computes hinge angles from the robot's state. Stateless between calls."""

    def __init__(
        self, genome: Sequence[float], scene: Scene, hidden: int, clock_hz: float
    ) -> None:
        self.scene = scene
        self.clock_hz = clock_hz
        self.w1, self.b1, self.w2, self.b2 = decode(
            genome, len(scene.hinge_qpos), hidden
        )
        start = scene.data.qpos[scene.core_qpos : scene.core_qpos + 2]
        self.start_distance = float(np.linalg.norm(scene.target_xy - start))
        if self.start_distance <= 0:
            raise ValueError("the target must not coincide with the spawn position")

    def inputs(self, data: mj.MjData) -> np.ndarray:
        """The network's inputs for the current simulation state, in this order:

            hinge angles ......... one per motor (the robot's sense of its posture)
            sin, cos of the clock  the rhythm to walk with
            cos, sin of bearing .. where the target is, relative to where the robot
                                   faces; positive bearing = target to the left
            distance ............. to the target, 1 at the start and 0 on arrival
        """
        scene = self.scene
        hinges = data.qpos[scene.hinge_qpos]

        phase = 2 * np.pi * self.clock_hz * data.time
        clock = (np.sin(phase), np.cos(phase))

        # Heading: the core's local +y axis (its FRONT face) projected on the ground.
        forward = data.xmat[scene.core_body].reshape(3, 3)[:2, 1]
        heading = np.arctan2(forward[1], forward[0])
        to_target = scene.target_xy - data.qpos[scene.core_qpos : scene.core_qpos + 2]
        bearing = np.arctan2(to_target[1], to_target[0]) - heading
        distance = np.linalg.norm(to_target) / self.start_distance

        return np.concatenate(
            [hinges, clock, (np.cos(bearing), np.sin(bearing)), (distance,)]
        )

    def __call__(self, data: mj.MjData) -> np.ndarray:
        """Hinge angle commands, in [-pi/2, pi/2]."""
        hidden = np.tanh(self.inputs(data) @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2) * HINGE_RANGE
