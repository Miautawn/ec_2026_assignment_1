"""Every parameter of the experiment, in one place.

To run something: edit `CONFIG` at the bottom, then run `run_experiment.py`.
Give each experiment its own `outputs("name")` folder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.config import BaseConfig

ASSIGNMENT_DIR: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = ASSIGNMENT_DIR / "outputs"


def outputs(name: str) -> dict[str, Path]:
    """Output folders for one named experiment: outputs/<name>/{results,figures}.

    Tables land beside them, in outputs/<name>/tables.
    """
    return {
        "results_dir": OUTPUTS_DIR / name / "results",
        "figures_dir": OUTPUTS_DIR / name / "figures",
    }


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    """Everything needed to reproduce the experiment.

    Inherited from `harness.config.BaseConfig` (set them here like any other):

        variants                        EAs to run, by registry name
        seeds = range(10)               One independent run per seed per variant
        population_size = 50            Starting individual count
        generations = 100               Number of iterations for 1 population
        offspring_per_generation = 50   How many offsprings to create each generation
                                        Eval budget = population_size + (generations * offsprings_per_gen)
        workers = None                  Runs in parallel; None = every core
        results_dir, figures_dir        Use **outputs("name")

    Attributes
    ----------
    world
        "flat", "rugged", "crater" or "amphitheatre" (see `scene.WORLDS`).
    world_params
        Overrides for the world's own parameters, e.g. {"height_of_noise": 0.15}.
    terrain_seed
        Seed for the terrain's random bumps. Fixed for the whole assignment and
        independent of the run seeds, so every run sees the same terrain.
    body
        Name of a John Set body, e.g. "gecko".
    spawn_position
        Where the robot is placed, (x, y, z). z is corrected to the terrain.
    target_y
        The target sits straight ahead of the spawn, at (spawn x, target_y):
        every body's core faces +y at spawn.
    sim_duration
        Simulated seconds per evaluation.
    hidden_size
        Neurons in the controller's single hidden layer.
    clock_hz
        Frequency of the sin/cos clock input.
    init_scale
        Standard deviation of the initial weights, N(0, init_scale). Also
        defines the random-search baseline, which samples the same way.
    control_hz
        How often the network is queried for new hinge angles; the physics
        runs at 500 Hz, so this must divide 500 evenly.
    """

    variants: tuple[str, ...] = ("random_search",)

    world: str = "crater"
    world_params: dict[str, Any] = field(default_factory=dict)
    terrain_seed: int = 0
    body: str = "gecko"
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.1)
    target_y: float = 3.0
    sim_duration: float = 15.0

    hidden_size: int = 8
    clock_hz: float = 1.0
    init_scale: float = 0.5
    control_hz: float = 50.0

    results_dir: Path = outputs("main")["results_dir"]
    figures_dir: Path = outputs("main")["figures_dir"]

    @property
    def spawn_xy(self) -> tuple[float, float]:
        return self.spawn_position[0], self.spawn_position[1]

    @property
    def target_xy(self) -> tuple[float, float]:
        return self.spawn_position[0], self.target_y


#: Quick end-to-end check: every code path, in well under a minute.
SMOKE = ExperimentConfig(
    variants=("random_search", "static_sigma", "self_adaptive"),
    seeds=(101, 420, 69),
    population_size=50,
    generations=50,
    offspring_per_generation=25,
    sim_duration=10.0,
    **outputs("smoke"),
)

#: The experiment `run_experiment.py` runs. Edit this, then re-run.
CONFIG = SMOKE
