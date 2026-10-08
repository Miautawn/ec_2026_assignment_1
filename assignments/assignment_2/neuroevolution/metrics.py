"""Per-generation numbers that explain *why* an EA behaves as it does.

Computed from the databases after a run and added as columns of
`per_generation.csv`; each function says what it measures. Numbers that do
not apply are left empty (random search has no parents, so no success rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from harness.dataset import Generation, population_after_selection

NAN = float("nan")


# --------------------------------------------------------------------------- #
#  Per individual: f(genotype, tags)
# --------------------------------------------------------------------------- #

def sigma(genotype: dict, tags: dict) -> float:  # noqa: ARG001
    """Mutation step size stored in the genome."""
    value = genotype.get("sigma")
    return NAN if value is None else float(value)


def distance(genotype: dict, tags: dict) -> float:  # noqa: ARG001
    """How far the robot travelled from its start."""
    return float(tags.get("distance_from_spawn", NAN))


# --------------------------------------------------------------------------- #
#  Per generation: f(Generation)
# --------------------------------------------------------------------------- #

def _mean_pairwise_distance(points: np.ndarray) -> float:
    return float(pdist(points).mean()) if len(points) >= 2 else NAN


def genotype_diversity(generation: Generation) -> float:
    """How different the genomes are: mean distance between every pair of weight vectors."""
    weights = np.array([g["weights"] for g in generation.population["genotype"]])
    return _mean_pairwise_distance(weights)


def behaviour_diversity(generation: Generation) -> float:
    """How different the robots' end positions are: mean distance between every pair."""
    ends = np.array([(t["final_x"], t["final_y"]) for t in generation.population["tags"]])
    return _mean_pairwise_distance(ends)


def _children(generation: Generation) -> pd.DataFrame:
    born = generation.born
    return born.loc[[t.get("origin") == "offspring" for t in born["tags"]]]


def _best_parent_fitness(children: pd.DataFrame, everyone: pd.DataFrame) -> np.ndarray:
    fitness = everyone["fitness_"].astype(float)
    return np.array([fitness.loc[t["parents"]].min() for t in children["tags"]])


def success_rate(generation: Generation) -> float:
    """Share of this generation's children that beat their best parent."""
    children = _children(generation)
    if children.empty:
        return NAN
    parents = _best_parent_fitness(children, generation.everyone)
    return float(np.mean(children["fitness_"].astype(float).to_numpy() < parents))


def improvement_over_parent(generation: Generation) -> float:
    """Mean of (best parent's fitness - child's fitness); positive means children improved."""
    children = _children(generation)
    if children.empty:
        return NAN
    parents = _best_parent_fitness(children, generation.everyone)
    return float(np.mean(parents - children["fitness_"].astype(float).to_numpy()))


def survival_rate(generation: Generation) -> float:
    """Share of this generation's children still present in the next one (empty for the last)."""
    children = _children(generation)
    if children.empty or generation.number == generation.last:
        return NAN
    return float((children["time_of_death"] > generation.number).mean())


def parent_fraction(generation: Generation) -> float:
    """Distinct parents of this generation's children, as a share of the population they came from."""
    children = _children(generation)
    # Parents are chosen from the previous generation's population.
    pool = population_after_selection(
        generation.everyone, generation.number - 1, generation.last
    )
    if children.empty or pool.empty:
        return NAN
    distinct = {parent for t in children["tags"] for parent in t["parents"]}
    return len(distinct) / len(pool)


def failure_rate(generation: Generation) -> float:
    """Share of this generation's evaluations in which the physics went unstable."""
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
