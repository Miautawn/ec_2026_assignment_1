"""
Single source of truth for every Assignment 1 parameter.

Budget, seeds and output paths come from `harness.config.BaseConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.config import BaseConfig

ASSIGNMENT_DIR: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    """Everything needed to reproduce the experiment.

    Attributes
    ----------
    num_modules
        max genome tree size

    crossover_rate
        probability to even do a crossover

    mutation_rate
        probability to even mutate genome

    tournament_size
        how many individuals participate in the tournament selection
    """

    variants: tuple[str, ...] = ("mutate_child", "mutate_parent", "random_search")

    num_modules: int = 20

    # Shared variation and selection settings for both EAs.
    crossover_rate: float = 0.9
    mutation_rate: float = 0.9
    tournament_size: int = 3

    results_dir: Path = ASSIGNMENT_DIR / "results"
    figures_dir: Path = ASSIGNMENT_DIR / "figures"


#: Quick end-to-end check: same code paths, ~30x less compute.
SMOKE = ExperimentConfig(
    seeds=(0, 1, 2),
    population_size=20,
    generations=10,
    offspring_per_generation=20,
    results_dir=ASSIGNMENT_DIR / "local-runs" / "smoke" / "results",
    figures_dir=ASSIGNMENT_DIR / "local-runs" / "smoke" / "figures",
)
