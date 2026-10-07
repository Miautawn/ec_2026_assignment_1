"""The EAs. To write a new one, subclass `NeuroEA`.

`NeuroEA` does the shared work (seeding, the initial population, evaluating
and logging), so an EA only lists its own steps in `stages()`. Below it:
random search, the baseline the brief requires, and two DRAFT EAs.
"""

# NOTE: deliberately NO `from __future__ import annotations` in this module.
# ARIEL's @EAOperation checks `params[0].annotation is Population` by identity;
# PEP 563 turns annotations into strings and every stage would be rejected.

from pathlib import Path

import numpy as np

from ariel.ec import EA, EAOperation, Individual, Population
from harness.variants import ea_settings, seed_everything

from .config import ExperimentConfig
from .evaluator import IS_MAXIMISATION, Evaluator
from .genotype import Genotype, from_individual, random_genotype, to_individual
from .provenance import mark_initial, mark_offspring, mark_random, record_evaluation


class NeuroEA(EA):
    """Base class for every Assignment 2 EA. Subclasses implement `stages()`.

        class MyEA(NeuroEA):
            def stages(self):
                return [EAOperation(self.reproduce), EAOperation(self.evaluate),
                        EAOperation(self.select_survivors)]

    Make children with `mark_offspring(child, parents, operators)` so they are
    logged, and use `self.cfg.population_size`, never ARIEL's global `config`.
    """

    #: σ stored on initial individuals. None: they were not made by mutation.
    #: An EA whose σ is a gene (self-adaptation) sets its starting value here.
    initial_sigma: float | None = None

    def __init__(self, seed: int, db_path: Path, cfg: ExperimentConfig) -> None:
        seed_everything(seed)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.evaluator = Evaluator(cfg)
        super().__init__(
            self.evaluate(self.initial_population()),
            operations=self.stages(),
            **ea_settings(db_path, cfg, is_maximisation=IS_MAXIMISATION),
        )

    def stages(self) -> list[EAOperation]:
        """The EA's pipeline, run once per generation, in order."""
        raise NotImplementedError

    # -- shared building blocks ---------------------------------------------- #

    def sample_genotype(self, sigma: float | None = None) -> Genotype:
        """Fresh random weights, N(0, init_scale)."""
        return random_genotype(
            self.rng, self.evaluator.genome_length, self.cfg.init_scale, sigma
        )

    def initial_population(self) -> Population:
        """The first `population_size` draws from this run's RNG.

        Drawn before anything else touches `self.rng`, so every EA started
        with the same seed gets the same initial weights.
        """
        return Population([
            mark_initial(to_individual(self.sample_genotype(self.initial_sigma)))
            for _ in range(self.cfg.population_size)
        ])

    def evaluate(self, population: Population) -> Population:
        """Score every unevaluated individual and record its provenance."""
        for individual in population.unevaluated:
            result = self.evaluator.evaluate(from_individual(individual).weights)
            record_evaluation(individual, result)
        return population


class RandomSearch(NeuroEA):
    """Random-search baseline, required by the brief at an identical budget.

    No selection and no inheritance: every generation the whole population is
    discarded and `offspring_per_generation` fresh controllers are sampled
    from the same distribution as the initial population. The fair curve
    against it is the cumulative best (`best_so_far`).
    """

    def stages(self) -> list[EAOperation]:
        return [EAOperation(self.resample), EAOperation(self.evaluate)]

    def resample(self, population: Population) -> Population:
        for individual in population:
            individual.alive = False
        population.extend(
            mark_random(to_individual(self.sample_genotype()))
            for _ in range(self.cfg.offspring_per_generation)
        )
        return population


# --------------------------------------------------------------------------- #
#  DRAFT EAs: written to test the pipeline, not tuned or validated.
#  To delete: remove everything below, and their two lines in variants.py.
# --------------------------------------------------------------------------- #

SIGMA = 0.1           # static σ, and the self-adaptive EA's starting σ
SIGMA_MIN = 1e-3      # self-adaptive floor (ε0): σ cannot collapse to zero
TOURNAMENT_SIZE = 3


class _MutationOnlyEA(NeuroEA):
    """(μ + λ): tournament parents, Gaussian mutation, no crossover; the best
    `population_size` of parents and children survive. Subclasses set σ."""

    def stages(self) -> list[EAOperation]:
        return [
            EAOperation(self.reproduce),
            EAOperation(self.evaluate),
            EAOperation(self.select_survivors),
        ]

    def next_sigma(self, parent_sigma: float | None) -> float:
        """The σ a child is mutated with."""
        raise NotImplementedError

    def tournament(self, candidates: list[Individual]) -> Individual:
        entrants = self.rng.integers(len(candidates), size=TOURNAMENT_SIZE)
        return min((candidates[i] for i in entrants), key=lambda ind: ind.fitness)

    def reproduce(self, population: Population) -> Population:
        parents = list(population)
        children = []
        for _ in range(self.cfg.offspring_per_generation):
            parent = self.tournament(parents)
            genotype = from_individual(parent)
            sigma = self.next_sigma(genotype.sigma)
            weights = genotype.weights + self.rng.normal(0.0, sigma, genotype.weights.size)
            child = to_individual(Genotype(weights, sigma))
            children.append(mark_offspring(child, [parent], ["mutation"]))
        population.extend(children)
        return population

    def select_survivors(self, population: Population) -> Population:
        ranked = sorted(population, key=lambda ind: ind.fitness)
        keep = {id(ind) for ind in ranked[: self.cfg.population_size]}
        for individual in population:
            individual.alive = id(individual) in keep
        return population


class StaticSigmaEA(_MutationOnlyEA):
    """DRAFT. Every child is mutated with the same, fixed σ (SIGMA)."""

    def next_sigma(self, parent_sigma: float | None) -> float:  # noqa: ARG002
        return SIGMA


class SelfAdaptiveEA(_MutationOnlyEA):
    """DRAFT. σ is a gene and evolves with the weights.

    Each child first mutates its parent's σ, σ' = max(σ · exp(τ · N(0, 1)),
    SIGMA_MIN) with τ = 1/sqrt(n), then uses σ' on its weights.
    """

    initial_sigma = SIGMA

    def next_sigma(self, parent_sigma: float | None) -> float:
        tau = 1.0 / np.sqrt(self.evaluator.genome_length)
        return max(parent_sigma * np.exp(tau * self.rng.normal()), SIGMA_MIN)
