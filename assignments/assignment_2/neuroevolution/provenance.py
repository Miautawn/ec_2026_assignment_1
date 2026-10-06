"""What every individual records about its own origin and evaluation.

Stored in the individual's ARIEL `tags` (a JSON column), so that any question
about what the operators did can be answered from the database after a run,
without re-running anything:

    origin               "initial" | "offspring" | "random"
    parents              database ids of its parents ([] if none)
    operators            variation operators that made it, in order applied
    final_x, final_y     where the evaluation ended   (from EvalResult)
    distance_from_spawn  how far it got, any direction (from EvalResult)
    failed               whether the physics was unstable (from EvalResult)

Only raw facts are recorded; statistics are computed later. Anything already
in the database is not duplicated: generation (time_of_birth), survival
(alive / time_of_death), fitness and σ (in the genotype). A parent's fitness or
σ is found by looking its id up.
"""

from __future__ import annotations

from collections.abc import Sequence

from ariel.ec import Individual

from .evaluator import EvalResult

ORIGINS = ("initial", "offspring", "random")

#: Every key an individual must carry once it has been created and evaluated.
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
