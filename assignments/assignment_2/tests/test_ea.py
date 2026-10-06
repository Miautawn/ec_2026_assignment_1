"""Any NeuroEA subclass runs through the shared harness with complete logging.

The EA here is a throwaway written for this test -- the point is that the kit
works for *whatever* EA we end up writing, not to pre-build one.
"""

# No `from __future__ import annotations`: this module defines EA stages.

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroevolution.config import ExperimentConfig
from neuroevolution.ea import NeuroEA, RandomSearch
from neuroevolution.genotype import Genotype, from_individual, to_individual
from neuroevolution.metrics import sigma
from neuroevolution.provenance import REQUIRED_TAGS, mark_offspring
from neuroevolution.variants import VARIANTS

from ariel.ec import EAOperation, Population
from harness import dataset, runner
from harness.variants import Variant


class ToyEA(NeuroEA):
    """Mutation-only (mu + lambda) with a fixed sigma. Exists only for tests."""

    initial_sigma = 0.1

    def stages(self):
        return [EAOperation(self.reproduce), EAOperation(self.evaluate),
                EAOperation(self.survive)]

    def reproduce(self, population: Population) -> Population:
        parents = list(population)
        children = []
        for _ in range(self.cfg.offspring_per_generation):
            parent = parents[self.rng.integers(len(parents))]
            g = from_individual(parent)
            child = Genotype(g.weights + self.rng.normal(0, g.sigma, g.weights.size), g.sigma)
            children.append(mark_offspring(to_individual(child), [parent], ["mutation"]))
        population.extend(children)
        return population

    def survive(self, population: Population) -> Population:
        keep = {id(i) for i in sorted(population, key=lambda i: i.fitness)[: self.cfg.population_size]}
        for individual in population:
            individual.alive = id(individual) in keep
        return population


REGISTRY = {**VARIANTS, "toy": Variant(ToyEA, "Toy EA (test only)", "#0072B2")}


@pytest.fixture
def cfg(tmp_path):
    return ExperimentConfig(
        variants=("random_search", "toy"), seeds=(0,),
        population_size=4, generations=2, offspring_per_generation=3,
        sim_duration=0.2,
        results_dir=tmp_path / "results", figures_dir=tmp_path / "figures",
    )


def rows(db_path):
    with sqlite3.connect(db_path) as db:
        records = db.execute(
            "SELECT id, genotype_, tags_, fitness_, time_of_birth FROM individual ORDER BY id"
        ).fetchall()
    return [
        {"id": i, "genotype": json.loads(g), "tags": json.loads(t), "fitness": f, "born": b}
        for i, g, t, f, b in records
    ]


@pytest.mark.parametrize("variant", ["random_search", "toy"])
def test_runs_through_the_harness_with_exact_budget_and_complete_provenance(variant, cfg):
    meta = runner.execute(variant, 0, cfg, REGISTRY)
    assert meta["evaluations"] == cfg.evaluation_budget == 10

    records = rows(cfg.db_path(variant, 0))
    ids = {r["id"] for r in records}
    for r in records:
        assert set(REQUIRED_TAGS) <= set(r["tags"])
        assert set(r["tags"]["parents"]) <= ids        # every parent is in the database
    origins = {r["tags"]["origin"] for r in records}
    assert origins == ({"initial", "random"} if variant == "random_search" else {"initial", "offspring"})


def test_harness_dataset_reads_a2_runs_including_sigma(cfg):
    for variant in cfg.variants:
        runner.execute(variant, 0, cfg, REGISTRY)
    tidy = dataset.load(cfg, {"sigma": sigma})
    by_variant = tidy.groupby("variant", observed=True)["mean_sigma"]
    assert by_variant.apply(lambda s: s.isna().all())["random_search"]   # no σ: no line
    assert by_variant.mean()["toy"] == pytest.approx(0.1)
    assert list(tidy.loc[tidy.variant == "toy", "evaluations_so_far"]) == [4, 7, 10]


def test_same_seed_reproduces_the_run_exactly(cfg, tmp_path):
    runner.execute("toy", 0, cfg, REGISTRY)
    first = rows(cfg.db_path("toy", 0))
    runner.execute("toy", 0, cfg, REGISTRY)
    assert rows(cfg.db_path("toy", 0)) == first


def test_every_variant_starts_from_the_same_weights_for_a_seed(cfg):
    for variant in cfg.variants:
        runner.execute(variant, 0, cfg, REGISTRY)
    initial = {
        variant: [r["genotype"]["weights"] for r in rows(cfg.db_path(variant, 0)) if r["born"] == 0]
        for variant in cfg.variants
    }
    assert initial["random_search"] == initial["toy"]


def test_random_search_never_inherits(cfg):
    runner.execute("random_search", 0, cfg, REGISTRY)
    for r in rows(cfg.db_path("random_search", 0)):
        assert r["tags"]["parents"] == []
        assert r["genotype"]["sigma"] is None


def test_an_ea_must_define_its_stages(cfg):
    with pytest.raises(NotImplementedError):
        NeuroEA(0, cfg.db_path("bare", 0), cfg)


def test_parallel_and_sequential_runs_are_identical(cfg, tmp_path):
    """The number of worker processes must never change a result."""
    from dataclasses import replace

    grid = replace(cfg, seeds=(0, 1))
    sequential = replace(grid, workers=1, results_dir=tmp_path / "seq" / "results")
    parallel = replace(grid, workers=2, results_dir=tmp_path / "par" / "results")
    runner.run_all(sequential, REGISTRY)
    runner.run_all(parallel, REGISTRY)
    for variant in grid.variants:
        for seed in grid.seeds:
            assert rows(parallel.db_path(variant, seed)) == rows(sequential.db_path(variant, seed))
