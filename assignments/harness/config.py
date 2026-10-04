"""The experiment parameters every assignment shares.

Each assignment subclasses `BaseConfig` to add its own parameters and to point
the output directories at itself. Subclasses must also be `frozen=True`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaseConfig:
    """Budget, seeds and output locations.

    Attributes
    ----------
    variants
        Names to run, resolved against the assignment's variant registry.
    seeds
        One independent run per seed per variant.
    population_size, generations, offspring_per_generation
        The evaluation budget is
        `population_size + generations * offspring_per_generation`. It must be
        identical across every variant, otherwise the comparison is unfair.
    results_dir, figures_dir
        Where runs and figures land. Tables go beside `results_dir`.
    """

    variants: tuple[str, ...] = ()
    seeds: tuple[int, ...] = tuple(range(10))

    population_size: int = 50
    generations: int = 100
    offspring_per_generation: int = 50

    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")

    @property
    def evaluation_budget(self) -> int:
        """Total fitness evaluations per run, identical across variants."""
        return self.population_size + self.generations * self.offspring_per_generation

    @property
    def tables_dir(self) -> Path:
        return self.results_dir.parent / "tables"

    def run_dir(self, variant: str, seed: int) -> Path:
        """Directory holding one run's database and metadata."""
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
