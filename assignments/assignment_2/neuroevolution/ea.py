"""The EA kit: everything an Assignment 2 EA needs, whatever it does.

A new EA subclasses `NeuroEA`, writes its own stages (selection, variation,
survivor selection -- any `Population -> Population` methods) and lists them
in `stages()`. The base class handles the rest:

  * seeding, and a seeded `self.rng` for all of the EA's own randomness;
  * one `Evaluator` (scene built once, reset per evaluation);
  * the initial population -- identical for every EA given the same seed,
    so runs can be compared pair-wise by seed;
  * the `evaluate` stage, which scores individuals AND records their fitness
    and provenance, so no EA can forget to log what happened;
  * the EA settings every variant must share (`is_maximisation=False`, ...).

    class MyEA(NeuroEA):
        def stages(self):
            return [EAOperation(self.reproduce), EAOperation(self.evaluate),
                    EAOperation(self.select_survivors)]

        def reproduce(self, population: Population) -> Population:
            ...  # make children with mark_offspring(child, parents, operators)

Use `self.cfg.population_size`, never ARIEL's global `config`, for the
population size: `EA` reads `target_population_size` from that global.
"""

# NOTE: deliberately NO `from __future__ import annotations` in this module.
# ARIEL's @EAOperation checks `params[0].annotation is Population` by identity;
# PEP 563 turns annotations into strings and every stage would be rejected.

from pathlib import Path

import numpy as np

from ariel.ec import EA, EAOperation, Population
from harness.variants import ea_settings, seed_everything

from .config import ExperimentConfig
from .evaluator import IS_MAXIMISATION, Evaluator
from .genotype import Genotype, from_individual, random_genotype, to_individual
from .provenance import mark_initial, mark_random, record_evaluation


class NeuroEA(EA):
    """Base class for every Assignment 2 EA. Subclasses implement `stages()`."""

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
