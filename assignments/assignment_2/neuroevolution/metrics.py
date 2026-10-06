"""The per-generation numbers that explain *why* an EA behaves as it does.

Computed from the database after a run (from the genotypes and the provenance
tags), and added as columns of `per_generation.csv` by the harness. They apply
to any EA; where a number does not apply it is empty (NaN) -- e.g. random
search has no parents, so it has no mutation success rate.

Per individual (harness adds `mean_<name>` and `best_<name>`):

    sigma        mutation step size stored in the genome
    distance     distance travelled from spawn

Per generation (harness adds one column each):

    genotype_diversity       mean pairwise distance between weight vectors of
                             the population present this generation
    behaviour_diversity      mean pairwise distance between their end positions
    success_rate             share of children born this generation that
                             beat their best parent (lower fitness)
    improvement_over_parent  mean of (best parent's fitness - child's fitness)
                             over those children; positive = children better
    survival_rate            share of those children still present in the
                             next generation (empty for the last generation)
    parent_fraction          distinct parents of this generation's children,
                             as a share of the population they were chosen from
    failure_rate             share of this generation's evaluations in which
                             the physics went unstable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from harness.dataset import Generation

NAN = float("nan")


# --------------------------------------------------------------------------- #
#  Per individual: f(genotype, tags)
# --------------------------------------------------------------------------- #

def sigma(genotype: dict, tags: dict) -> float:  # noqa: ARG001
    value = genotype.get("sigma")
    return NAN if value is None else float(value)


def distance(genotype: dict, tags: dict) -> float:  # noqa: ARG001
    return float(tags.get("distance_from_spawn", NAN))


# --------------------------------------------------------------------------- #
#  Per generation: f(Generation)
# --------------------------------------------------------------------------- #

def _mean_pairwise_distance(points: np.ndarray) -> float:
    return float(pdist(points).mean()) if len(points) >= 2 else NAN


def genotype_diversity(generation: Generation) -> float:
    weights = np.array([g["weights"] for g in generation.alive["genotype"]])
    return _mean_pairwise_distance(weights)


def behaviour_diversity(generation: Generation) -> float:
    ends = np.array([(t["final_x"], t["final_y"]) for t in generation.alive["tags"]])
    return _mean_pairwise_distance(ends)


def _children(generation: Generation) -> pd.DataFrame:
    born = generation.born
    return born.loc[[t.get("origin") == "offspring" for t in born["tags"]]]


def _best_parent_fitness(children: pd.DataFrame, everyone: pd.DataFrame) -> np.ndarray:
    fitness = everyone["fitness_"].astype(float)
    return np.array([fitness.loc[t["parents"]].min() for t in children["tags"]])


def success_rate(generation: Generation) -> float:
    children = _children(generation)
    if children.empty:
        return NAN
    parents = _best_parent_fitness(children, generation.everyone)
    return float(np.mean(children["fitness_"].astype(float).to_numpy() < parents))


def improvement_over_parent(generation: Generation) -> float:
    children = _children(generation)
    if children.empty:
        return NAN
    parents = _best_parent_fitness(children, generation.everyone)
    return float(np.mean(parents - children["fitness_"].astype(float).to_numpy()))


def survival_rate(generation: Generation) -> float:
    children = _children(generation)
    if children.empty or generation.number == generation.last:
        return NAN
    return float((children["time_of_death"] > generation.number).mean())


def parent_fraction(generation: Generation) -> float:
    children = _children(generation)
    pool = generation.alive.loc[generation.alive["time_of_birth"] < generation.number]
    if children.empty or pool.empty:
        return NAN
    distinct = {parent for t in children["tags"] for parent in t["parents"]}
    return len(distinct) / len(pool)


def failure_rate(generation: Generation) -> float:
    born = generation.born
    if born.empty:
        return NAN
    return float(np.mean([bool(t.get("failed")) for t in born["tags"]]))


METRICS = {"sigma": sigma, "distance": distance}

POPULATION_METRICS = {
    "genotype_diversity": genotype_diversity,
    "behaviour_diversity": behaviour_diversity,
    "success_rate": success_rate,
    "improvement_over_parent": improvement_over_parent,
    "survival_rate": survival_rate,
    "parent_fraction": parent_fraction,
    "failure_rate": failure_rate,
}
