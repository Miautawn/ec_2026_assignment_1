"""
Single source of truth for every experiment parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ASSIGNMENT_DIR: Path = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce the experiment.

    Attributes
    ----------
    variants
        Names to run, resolved against `variants.VARIANTS`. The random-search
        baseline is mandatory per the assignment brief, so keep it in the list.
    seeds
        One independent evoluation run per seed per variant.
    population_size, generations, offspring_per_generation
        The evaluation budget is
        `population_size + generations * offspring_per_generation`, and it is
        held *identical* across every variant experiment, otherwise
        the comparisons will be unfair

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
    seeds: tuple[int, ...] = tuple(range(10))

    population_size: int = 50
    generations: int = 100
    offspring_per_generation: int = 50

    num_modules: int = 20

    # IDK these might be hardcoded in the AE variants
    # but for now let's keep them here!
    crossover_rate: float = 0.9
    mutation_rate: float = 0.9
    tournament_size: int = 3

    results_dir: Path = ASSIGNMENT_DIR / "results"
    figures_dir: Path = ASSIGNMENT_DIR / "figures"

    @property
    def evaluation_budget(self) -> int:
        """Total fitness evaluations per run, identical across variants."""
        return self.population_size + self.generations * self.offspring_per_generation

    def run_dir(self, variant: str, seed: int) -> Path:
        """Directory holding one run's database and metadata.
        """
        return self.results_dir / variant / f"seed_{seed:02d}"

    def db_path(self, variant: str, seed: int) -> Path:
        return self.run_dir(variant, seed) / "database.db"

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot, written beside every run."""
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = list(value)
        out["evaluation_budget"] = self.evaluation_budget
        return out


#: Quick end-to-end check: same code paths, ~30x less compute.
SMOKE = ExperimentConfig(
    seeds=(0, 1, 2),
    population_size=20,
    generations=10,
    offspring_per_generation=20,
)
