"""Mutation-order variants and an independent random-search baseline."""

# NOTE: deliberately NO `from __future__ import annotations` in this module.
# ARIEL's @EAOperation validates a stage with
# `params[0].annotation is not Population`, an identity check against the real
# class. PEP 563 turns annotations into strings, so with the future import
# every stage fails with the baffling
#   "first argument must be annotated as Population, got 'Population'".
# Any module defining EA stages must leave the future import out.

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.ec.genotypes.tree.operators import (
    add_node,
    crossover_subtree,
    mutate_replace_node,
    random_tree,
    remove_subtree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

from .config import ExperimentConfig
from .fitness import IS_MAXIMISATION, evaluate_body


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    """Seed every RNG ARIEL may touch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_individual(cfg: ExperimentConfig) -> Individual:
    """Draw a target size uniformly, counting the core as one module."""
    genome = random_tree(max_modules=random.randint(1, cfg.num_modules) - 1)
    individual = Individual()
    individual.genotype = genome.to_dict()
    return individual


def mutate(genome: TreeGenome, cfg: ExperimentConfig) -> TreeGenome:
    """Try one equally likely add, remove-subtree or replace-node mutation."""
    child = deepcopy(genome)
    if random.random() >= cfg.mutation_rate:
        return child
    operation = random.choice(("add", "remove", "replace"))
    if operation == "add" and len(child.nodes) < cfg.num_modules:
        free = [
            (nid, face.name)
            for nid, data in child.nodes.items()
            for face in ALLOWED_FACES[ModuleType[data["type"]]]
            if not any(
                e["parent"] == nid and e["face"] == face.name for e in child.edges
            )
        ]
        if free:
            parent, face = random.choice(free)
            kind = random.choice((ModuleType.BRICK, ModuleType.HINGE))
            rotation = random.choice(ALLOWED_ROTATIONS[kind]).name
            add_node(child, parent, face, max(child.nodes) + 1, kind.name, rotation)
    elif operation == "remove":
        candidates = [nid for nid in child.nodes if nid != IDX_OF_CORE]
        if candidates:
            remove_subtree(child, random.choice(candidates))
    elif operation == "replace":
        mutate_replace_node(child)
    validate_genome_dict(child.to_dict())
    return child


def crossover(
    a: TreeGenome, b: TreeGenome, cfg: ExperimentConfig
) -> tuple[TreeGenome, TreeGenome]:
    """Exchange non-core subtrees; reject the pair if either exceeds the cap."""
    if random.random() >= cfg.crossover_rate:
        return deepcopy(a), deepcopy(b)
    children = crossover_subtree(a, b)
    if any(len(child.nodes) > cfg.num_modules for child in children):
        return deepcopy(a), deepcopy(b)
    for child in children:
        validate_genome_dict(child.to_dict())
    return children


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
        "first_generation_id": 0,
        "is_maximisation": IS_MAXIMISATION,  # fitness is lower-is-better
        "db_file_path": db_path,
        "db_handling": "delete",  # the runner guarantees a clean directory
        "quiet": True,
    }


# --------------------------------------------------------------------------- #
#  Algorithms
# --------------------------------------------------------------------------- #


class RandomSearch(EA):
    """Independent random samples, with an elite archive for reporting."""

    def __init__(self, seed: int, db_path: Path, cfg: ExperimentConfig) -> None:
        for name in (
            "population_size",
            "offspring_per_generation",
            "num_modules",
            "tournament_size",
        ):
            value = getattr(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(cfg.generations, int)
            or isinstance(cfg.generations, bool)
            or cfg.generations < 0
        ):
            raise ValueError("generations must be a nonnegative integer")
        for name in ("mutation_rate", "crossover_rate"):
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
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
        ranked = sorted(population, key=lambda ind: ind.fitness)
        survivors = {id(ind) for ind in ranked[: self.cfg.population_size]}
        for individual in population:
            individual.alive = id(individual) in survivors
        return population


class _MutationOrderEA(RandomSearch):
    """Tournament mating and elitist (mu + lambda) survivor selection."""

    mutate_before_crossover = False

    def select_parent(self, parents: list[Individual]) -> TreeGenome:
        winner = min(
            random.choices(parents, k=self.cfg.tournament_size),
            key=lambda ind: ind.fitness,
        )
        return TreeGenome.from_dict(deepcopy(winner.genotype))

    def propose(self, population: Population) -> Population:
        parents = list(population)
        offspring = []
        while len(offspring) < self.cfg.offspring_per_generation:
            a, b = self.select_parent(parents), self.select_parent(parents)
            if self.mutate_before_crossover:
                a, b = mutate(a, self.cfg), mutate(b, self.cfg)
            a, b = crossover(a, b, self.cfg)
            if not self.mutate_before_crossover:
                a, b = mutate(a, self.cfg), mutate(b, self.cfg)
            for genome in (a, b):
                if len(offspring) == self.cfg.offspring_per_generation:
                    break
                child = Individual()
                child.genotype = genome.to_dict()
                offspring.append(child)
        population.extend(offspring)
        return population


class MutateChildEA(_MutationOrderEA):
    """Select parents, cross over, then mutate each child."""


class MutateParentEA(_MutationOrderEA):
    """Select parents, mutate copies, then cross over."""

    mutate_before_crossover = True


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #

#: name -> EA subclass.
VARIANTS = {
    "mutate_child": MutateChildEA,
    "mutate_parent": MutateParentEA,
    "random_search": RandomSearch,
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
