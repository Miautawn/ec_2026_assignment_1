"""Genotype storage, validation and random initialisation."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroevolution.config import ExperimentConfig
from neuroevolution.evaluator import Evaluator
from neuroevolution.genotype import (
    Genotype,
    from_individual,
    random_genotype,
    to_individual,
)

from ariel.ec import Individual


def test_database_round_trip_is_exact_and_rescores_identically(tmp_path):
    """What the EA stores is exactly what a replay reads back."""
    evaluator = Evaluator(ExperimentConfig(sim_duration=1.0))
    original = random_genotype(
        np.random.default_rng(3), evaluator.genome_length, 0.5, sigma=0.1
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(to_individual(original))
        session.commit()
    with Session(engine) as session:
        restored = from_individual(session.exec(select(Individual)).one())

    assert np.array_equal(restored.weights, original.weights)  # bit-for-bit
    assert restored.sigma == original.sigma
    assert evaluator.evaluate(restored.weights) == evaluator.evaluate(original.weights)


def test_random_genotype_is_seeded_and_scaled():
    a = random_genotype(np.random.default_rng(0), 150, 0.5)
    b = random_genotype(np.random.default_rng(0), 150, 0.5)
    assert np.array_equal(a.weights, b.weights)
    assert a.weights.shape == (150,)
    assert a.sigma is None
    big = random_genotype(np.random.default_rng(1), 100_000, 0.5)
    assert big.weights.std() == pytest.approx(0.5, rel=0.02)


def test_weights_are_read_only_and_copied():
    source = np.zeros(4)
    genotype = Genotype(source, 0.2)
    source[0] = 99.0                      # changing the input does not leak in
    assert genotype.weights[0] == 0.0
    with pytest.raises(ValueError):
        genotype.weights[0] = 1.0         # and the genome itself cannot be edited


@pytest.mark.parametrize(
    "weights,sigma",
    [
        ([], None),
        ([[1.0, 2.0]], None),
        ([1.0, float("nan")], None),
        ([1.0], 0.0),
        ([1.0], -0.1),
        ([1.0], float("inf")),
    ],
)
def test_invalid_genotypes_are_rejected(weights, sigma):
    with pytest.raises(ValueError):
        Genotype(np.asarray(weights, dtype=float), sigma)
