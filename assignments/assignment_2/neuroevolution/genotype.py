"""What a genome is: the network's weights plus the mutation step size σ.

Stored on each individual as {"weights": [...], "sigma": s} and read back
bit-for-bit, so a stored robot always replays exactly the same.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ariel.ec import Individual


@dataclass(frozen=True, eq=False)
class Genotype:
    """One genome: the weights, and the σ that created it.

    `sigma` is the mutation step size: the standard deviation of the noise added
    to every weight. One σ for the whole vector. For the self-adaptive EA it is a
    gene that evolves; for other EAs it records the σ used; None for individuals
    not made by mutation (random search). Weights are read-only, so making a
    child can never change its parent by accident.
    """
    weights: np.ndarray
    sigma: float | None = None

    def __post_init__(self) -> None:
        weights = np.array(self.weights, dtype=float)  # always a private copy
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("weights must be a non-empty flat vector")
        if not np.isfinite(weights).all():
            raise ValueError("weights must be finite")
        if self.sigma is not None and not (np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError("sigma must be a positive finite number, or None")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        if self.sigma is not None:
            object.__setattr__(self, "sigma", float(self.sigma))

    def to_json(self) -> dict[str, Any]:
        return {"weights": self.weights.tolist(), "sigma": self.sigma}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Genotype:
        return cls(np.asarray(payload["weights"], dtype=float), payload["sigma"])


def random_genotype(
    rng: np.random.Generator,
    length: int,
    init_scale: float,
    sigma: float | None = None,
) -> Genotype:
    """Weights drawn from N(0, init_scale).

    Used for every initial population AND for random search, so this
    distribution also defines the random-search baseline.
    """
    return Genotype(rng.normal(0.0, init_scale, size=length), sigma)


def to_individual(genotype: Genotype) -> Individual:
    individual = Individual()
    individual.genotype = genotype.to_json()
    return individual


def from_individual(individual: Individual) -> Genotype:
    return Genotype.from_json(individual.genotype)
