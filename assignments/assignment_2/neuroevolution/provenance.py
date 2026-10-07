"""What each individual writes down about itself: its parents, how it was made,
and what it did.

Stored in its tags, so after a run we can see what the operators actually did
(e.g. how often children beat their parents) without re-running anything.
"""

from __future__ import annotations

from collections.abc import Sequence

from ariel.ec import Individual

from .evaluator import EvalResult

ORIGINS = ("initial", "offspring", "random")

#: Every tag an individual carries once created and evaluated:
#:   origin               "initial" | "offspring" | "random"
#:   parents              database ids of its parents ([] if none)
#:   operators            variation operators that made it, in order
#:   final_x, final_y     where its evaluation ended
#:   distance_from_spawn  how far it got, in any direction
#:   failed               whether the physics went unstable
#: Facts already in the database (birth, survival, fitness, σ) are not repeated.
REQUIRED_TAGS = (
    "origin", "parents", "operators",
    "final_x", "final_y", "distance_from_spawn", "failed",
)


def mark_initial(individual: Individual) -> Individual:
    """A member of a run's initial population."""
    individual.tags = {"origin": "initial", "parents": [], "operators": []}
    return individual


def mark_random(individual: Individual) -> Individual:
    """A fresh random sample (random search), with no parents."""
    individual.tags = {"origin": "random", "parents": [], "operators": []}
    return individual


def mark_offspring(
    individual: Individual,
    parents: Sequence[Individual],
    operators: Sequence[str],
) -> Individual:
    """A child made from `parents` by `operators`, in the order applied.

    Raises
    ------
    ValueError
        If there are no parents or operators, or a parent has no database id
        yet (parents must come from an earlier, committed generation).
    """
    if not parents:
        raise ValueError("an offspring needs at least one parent")
    if not operators:
        raise ValueError("an offspring needs at least one operator")
    if any(parent.id is None for parent in parents):
        raise ValueError("every parent must already be stored (have a database id)")
    individual.tags = {
        "origin": "offspring",
        "parents": [int(parent.id) for parent in parents],
        "operators": list(operators),
    }
    return individual


def record_evaluation(individual: Individual, result: EvalResult) -> Individual:
    """Store an evaluation's fitness and behaviour on the individual."""
    individual.fitness = result.fitness
    individual.tags = result.as_tags()
    return individual


def missing_tags(individual: Individual) -> list[str]:
    """Required keys this individual lacks -- empty when complete."""
    return [key for key in REQUIRED_TAGS if key not in individual.tags]
