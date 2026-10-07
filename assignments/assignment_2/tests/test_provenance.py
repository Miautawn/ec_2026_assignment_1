"""Provenance tags: complete, correct, and readable back from the database."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroevolution.config import ExperimentConfig
from neuroevolution.evaluator import Evaluator
from neuroevolution.genotype import random_genotype, to_individual
from neuroevolution.provenance import (
    mark_initial,
    mark_offspring,
    mark_random,
    missing_tags,
    record_evaluation,
)

from ariel.ec import Individual


@pytest.fixture(scope="module")
def evaluator():
    return Evaluator(ExperimentConfig(sim_duration=1.0))


def new_individual(evaluator, seed):
    rng = np.random.default_rng(seed)
    return to_individual(random_genotype(rng, evaluator.genome_length, 0.5, sigma=0.1))


def test_lineage_survives_the_database_and_links_child_to_parent(evaluator, tmp_path):
    """A child's parent ids lead back to the parent's stored fitness."""
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    SQLModel.metadata.create_all(engine)

    parent = mark_initial(new_individual(evaluator, 0))
    record_evaluation(parent, evaluator.evaluate(parent.genotype["weights"]))
    with Session(engine) as session:
        session.add(parent)
        session.commit()
        session.refresh(parent)
        parent_id, parent_fitness = parent.id, parent.fitness

    child = mark_offspring(new_individual(evaluator, 1), [parent], ["mutation"])
    record_evaluation(child, evaluator.evaluate(child.genotype["weights"]))
    with Session(engine) as session:
        session.add(child)
        session.commit()

    with Session(engine) as session:
        rows = {ind.id: ind for ind in session.exec(select(Individual))}
    stored_child = next(ind for ind in rows.values() if ind.tags["origin"] == "offspring")

    assert missing_tags(stored_child) == []
    assert stored_child.tags["parents"] == [parent_id]
    assert stored_child.tags["operators"] == ["mutation"]
    assert rows[parent_id].fitness == parent_fitness   # "did it beat its parent?" is answerable
    assert stored_child.tags["failed"] is False


def test_each_origin_is_complete_after_evaluation(evaluator):
    parent = mark_initial(new_individual(evaluator, 0))
    parent.id = 7                                      # as if already stored
    for individual in (
        mark_initial(new_individual(evaluator, 1)),
        mark_random(new_individual(evaluator, 2)),
        mark_offspring(new_individual(evaluator, 3), [parent, parent], ["crossover", "mutation"]),
    ):
        assert missing_tags(individual)                # not complete before evaluation
        record_evaluation(individual, evaluator.evaluate(individual.genotype["weights"]))
        assert missing_tags(individual) == []
        assert not individual.requires_eval


def test_evaluation_keeps_the_origin_tags(evaluator):
    individual = mark_random(new_individual(evaluator, 4))
    result = evaluator.evaluate(individual.genotype["weights"])
    record_evaluation(individual, result)
    assert individual.tags["origin"] == "random"
    assert individual.fitness == result.fitness
    assert individual.tags["distance_from_spawn"] == result.distance_from_spawn


@pytest.mark.parametrize(
    "parents,operators,message",
    [
        ([], ["mutation"], "at least one parent"),
        (["stored"], [], "at least one operator"),
        (["unstored"], ["mutation"], "database id"),
    ],
)
def test_invalid_offspring_are_rejected(parents, operators, message, evaluator):
    stored, unstored = Individual(), Individual()
    stored.id = 1
    lookup = {"stored": stored, "unstored": unstored}
    with pytest.raises(ValueError, match=message):
        mark_offspring(Individual(), [lookup[p] for p in parents], operators)
