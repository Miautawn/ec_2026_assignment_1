"""Run from the repository root with uv run pytest assignments/assignment_1/tests."""

import json
import sqlite3
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import dataset, variants
from experiment.config import ExperimentConfig
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict


@pytest.fixture
def cfg():
    return ExperimentConfig(
        population_size=4, offspring_per_generation=3, generations=2, num_modules=8
    )


def rows(path):
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT genotype_, fitness_, alive, time_of_birth, time_of_death, requires_eval "
            "FROM individual ORDER BY id"
        ).fetchall()


@pytest.mark.parametrize("name", variants.VARIANTS)
def test_budget_reproducibility_and_database(name, cfg, tmp_path, monkeypatch):
    calls = []
    real_fitness = variants.evaluate_body

    def counted(body):
        calls.append(1)
        return real_fitness(body)

    monkeypatch.setattr(variants, "evaluate_body", counted)
    for repeat in range(2):
        path = tmp_path / f"{repeat}.db"
        algorithm = variants.build(name, 7, path, cfg)
        algorithm.run()
        algorithm.engine.dispose()
        records = rows(path)
        assert len(records) == cfg.evaluation_budget
        assert sum(row[2] for row in records) == cfg.population_size
        assert all(row[5] == 0 for row in records)
        for genotype, *_ in records:
            payload = json.loads(genotype)
            validate_genome_dict(payload)
            assert 1 <= len(payload["nodes"]) <= cfg.num_modules
        summary = dataset.summarise_run(path, name, 7)
        assert list(summary.evaluations_so_far) == [4, 7, 10]
        assert summary.best_so_far.diff().dropna().le(0).all()
    assert len(calls) == 2 * cfg.evaluation_budget
    assert rows(tmp_path / "0.db") == rows(tmp_path / "1.db")


@pytest.mark.parametrize(
    "name,expected",
    [("mutate_child", ["c", "m", "m"]), ("mutate_parent", ["m", "m", "c"])],
)
def test_order_and_parent_copying(name, expected, cfg, tmp_path, monkeypatch):
    cfg = replace(cfg, offspring_per_generation=2)
    algorithm = variants.build(name, 0, tmp_path / "order.db", cfg)
    algorithm.fetch_population()
    population = algorithm.population
    before = deepcopy([ind.genotype for ind in population])
    events = []

    def mutation(genome, cfg):
        events.append("m")
        genome.nodes[0]["rotation"] = "DEG_45"
        return genome

    def cross(a, b, cfg):
        events.append("c")
        return a, b

    monkeypatch.setattr(variants, "mutate", mutation)
    monkeypatch.setattr(variants, "crossover", cross)
    algorithm.propose(population)
    assert events == expected
    assert [ind.genotype for ind in population[: cfg.population_size]] == before
    assert all(ind.requires_eval for ind in population[cfg.population_size :])
    assert len({id(ind) for ind in population}) == len(population)
    algorithm.engine.dispose()


def test_operators_preserve_validity_and_can_change_shapes(cfg):
    cfg = replace(cfg, mutation_rate=1, crossover_rate=1)
    variants.seed_everything(42)
    crossed = mutated = False
    for _ in range(100):
        a = TreeGenome.from_dict(variants.random_individual(cfg).genotype)
        b = TreeGenome.from_dict(variants.random_individual(cfg).genotype)
        saved = deepcopy((a.to_dict(), b.to_dict()))
        children = variants.crossover(a, b, cfg)
        for original, child in zip((a, b), children):
            crossed |= not nx.is_isomorphic(original.to_networkx(), child.to_networkx())
            result = variants.mutate(child, cfg)
            mutated |= not nx.is_isomorphic(child.to_networkx(), result.to_networkx())
            validate_genome_dict(result.to_dict())
            assert 1 <= len(result.nodes) <= cfg.num_modules
        assert (a.to_dict(), b.to_dict()) == saved
    assert crossed and mutated


def test_zero_rates_return_independent_copies(cfg):
    cfg = replace(cfg, mutation_rate=0, crossover_rate=0)
    variants.seed_everything(12)
    a = TreeGenome.from_dict(variants.random_individual(cfg).genotype)
    b = TreeGenome.from_dict(variants.random_individual(cfg).genotype)
    c, d = variants.crossover(a, b, cfg)
    assert (c.to_dict(), d.to_dict()) == (a.to_dict(), b.to_dict())
    assert c is not a and d is not b
    changed = variants.mutate(a, cfg)
    assert changed.to_dict() == a.to_dict()
    assert changed is not a


def test_all_variants_start_identically(cfg, tmp_path):
    initial = []
    for name in variants.VARIANTS:
        path = tmp_path / f"{name}.db"
        algorithm = variants.build(name, 19, path, cfg)
        initial.append(rows(path))
        algorithm.engine.dispose()
    assert initial[0] == initial[1] == initial[2]


def test_random_search_does_not_use_parents(cfg, tmp_path):
    algorithm = variants.build("random_search", 0, tmp_path / "random.db", cfg)
    algorithm.fetch_population()
    parents = algorithm.population
    variants.seed_everything(9)
    expected = [
        variants.random_individual(cfg).genotype
        for _ in range(cfg.offspring_per_generation)
    ]
    variants.seed_everything(9)
    algorithm.propose(parents)
    assert [ind.genotype for ind in parents[cfg.population_size :]] == expected
    algorithm.engine.dispose()


@pytest.mark.parametrize("name", variants.VARIANTS)
def test_single_module_zero_generations(name, cfg, tmp_path):
    cfg = replace(cfg, num_modules=1, generations=0)
    algorithm = variants.build(name, 0, tmp_path / "zero.db", cfg)
    algorithm.run()
    assert len(rows(tmp_path / "zero.db")) == cfg.population_size
    algorithm.engine.dispose()


@pytest.mark.parametrize(
    "field,value",
    [
        ("population_size", 0),
        ("num_modules", 0),
        ("generations", -1),
        ("mutation_rate", 1.1),
        ("crossover_rate", -0.1),
        ("tournament_size", 0),
    ],
)
def test_bad_config_rejected(field, value, cfg, tmp_path):
    with pytest.raises(ValueError):
        variants.build(
            "mutate_child", 0, tmp_path / "bad.db", replace(cfg, **{field: value})
        )
