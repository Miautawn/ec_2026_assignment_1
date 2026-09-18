"""
EA algorithm registry for the experiment
TODO: update with actual variants when they are implemented!
"""

# NOTE: deliberately NO `from __future__ import annotations` in this module.
# ARIEL's @EAOperation validates a stage with
# `params[0].annotation is not Population`, an identity check against the real
# class. PEP 563 turns annotations into strings, so with the future import
# every stage fails with the baffling
#   "first argument must be annotated as Population, got 'Population'".
# Any module defining EA stages must leave the future import out.

import random
from pathlib import Path

import numpy as np
import torch

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from .config import ExperimentConfig
from .fitness import IS_MAXIMISATION, evaluate_body


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #

def seed_everything(seed: int) -> None:
    """Seed every RNG ARIEL may touch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_individual(cfg: ExperimentConfig) -> Individual:
    """A random body that decodes to at least one module."""
    while True:
        genome = random_tree(max_modules=cfg.num_modules)
        if genome.to_networkx().number_of_nodes() > 0:
            individual = Individual()
            individual.genotype = genome.to_dict()
            return individual


def evaluate(population: Population) -> Population:
    """Score every unevaluated individual. A shared EA stage."""
    for individual in population.unevaluated:
        body = TreeGenome.from_dict(individual.genotype).to_networkx()
        individual.fitness = evaluate_body(body)
    return population


def _initial(cfg: ExperimentConfig) -> Population:
    return evaluate(
        Population([random_individual(cfg) for _ in range(cfg.population_size)])
    )


def _ea_kwargs(db_path: Path, cfg: ExperimentConfig) -> dict:
    """Settings every variant must share for the comparison to be valid."""
    return {
        "num_steps": cfg.generations,
        "is_maximisation": IS_MAXIMISATION,  # fitness is lower-is-better
        "db_file_path": db_path,
        "db_handling": "delete",  # the runner guarantees a clean directory
        "quiet": True,
    }


# --------------------------------------------------------------------------- #
#  Temporary variants
#  TODO: replace with whatever we develop later
# --------------------------------------------------------------------------- #

class StubEA(EA):
    """
    TODO: Remove me when real classes arrive!

    Proposes fresh random bodies each generation and keeps the best
    `population_size` seen so far. Enough to produce a descending curve so the
    figures can be checked, and nothing more.
    """

    def __init__(self, seed: int, db_path: Path, cfg: ExperimentConfig) -> None:
        seed_everything(seed)
        self.cfg = cfg
        super().__init__(
            _initial(cfg),
            operations=[
                EAOperation(self.propose),
                EAOperation(evaluate),
                EAOperation(self.keep_best),
            ],
            **_ea_kwargs(db_path, cfg),
        )

    def propose(self, population: Population) -> Population:
        population.extend(
            random_individual(self.cfg)
            for _ in range(self.cfg.offspring_per_generation)
        )
        return population

    def keep_best(self, population: Population) -> Population:
        ranked = sorted(
            (ind for ind in population if ind.fitness_ is not None),
            key=lambda ind: ind.fitness,
        )
        survivors = {id(ind) for ind in ranked[: self.cfg.population_size]}
        for individual in population:
            individual.alive = id(individual) in survivors
        return population

# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #

#: name -> EA subclass.
VARIANTS = {
    "mutate_child": StubEA,
    "mutate_parent": StubEA,
    "random_search": StubEA,
}

#: One label and one colour per variant, so every figure reads as a set.
VARIANT_LABELS = {
    "mutate_child": "Mutate offspring (after crossover)",
    "mutate_parent": "Mutate parents (before crossover)",
    "random_search": "Random search (baseline)",
}

VARIANT_COLOURS = {
    "mutate_child": "#0072B2",
    "mutate_parent": "#D55E00",
    "random_search": "#7F7F7F",
}


def build(variant: str, seed: int, db_path: Path, cfg: ExperimentConfig) -> EA:
    return VARIANTS[variant](seed=seed, db_path=db_path, cfg=cfg)


def label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def colour(variant: str) -> str:
    return VARIANT_COLOURS.get(variant, "#333333")
