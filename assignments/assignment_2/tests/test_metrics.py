"""Per-generation metrics: definitions on hand-built data, then end to end."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroevolution import metrics
from neuroevolution.config import ExperimentConfig
from neuroevolution.metrics import METRICS, POPULATION_METRICS

from harness import dataset, runner
from harness.dataset import Generation


def individual(id_, fitness, born, died, *, origin="initial", parents=(),
               weights=(0.0, 0.0), end=(0.0, 0.0), failed=False, sigma=None):
    return {
        "id": id_, "fitness_": fitness, "time_of_birth": born, "time_of_death": died,
        "genotype": {"weights": list(weights), "sigma": sigma},
        "tags": {"origin": origin, "parents": list(parents), "final_x": end[0],
                 "final_y": end[1], "failed": failed, "distance_from_spawn": 0.0},
    }


def generation(rows, number=1, last=5):
    frame = pd.DataFrame(rows)
    alive = frame[(frame.time_of_birth <= number) & (frame.time_of_death >= number)]
    born = frame[frame.time_of_birth == number]
    return Generation(number, last, alive, born, frame.set_index("id", drop=False))


# Two parents (fitness 5 and 3) and two children, generation 1.
FAMILY = [
    individual(1, 5.0, 0, 1),
    individual(2, 3.0, 0, 3),
    individual(3, 2.0, 1, 2, origin="offspring", parents=[1, 2]),  # beats best parent (3)
    individual(4, 4.0, 1, 1, origin="offspring", parents=[1, 2]),  # beats parent 1 only
]


def test_success_counts_beating_the_best_parent_only():
    assert metrics.success_rate(generation(FAMILY)) == 0.5


def test_improvement_is_measured_against_the_best_parent():
    # (3 - 2) and (3 - 4) -> mean 0
    assert metrics.improvement_over_parent(generation(FAMILY)) == 0.0


def test_survival_into_the_next_generation():
    assert metrics.survival_rate(generation(FAMILY)) == 0.5   # child 3 survives, child 4 not
    assert math.isnan(metrics.survival_rate(generation(FAMILY, last=1)))


def test_parent_fraction():
    pool = FAMILY + [individual(5, 9.0, 0, 1), individual(6, 9.0, 0, 1)]
    assert metrics.parent_fraction(generation(pool)) == 0.5   # parents {1, 2} of a pool of 4


def test_diversity_is_mean_pairwise_distance():
    triangle = [  # a 3-4-5 triangle: pairwise distances 3, 4, 5
        individual(1, 1.0, 0, 1, weights=(0, 0), end=(0, 0)),
        individual(2, 1.0, 0, 1, weights=(3, 0), end=(3, 0)),
        individual(3, 1.0, 0, 1, weights=(0, 4), end=(0, 4)),
    ]
    assert metrics.genotype_diversity(generation(triangle)) == pytest.approx(4.0)
    assert metrics.behaviour_diversity(generation(triangle)) == pytest.approx(4.0)
    clones = [individual(i, 1.0, 0, 1, weights=(1, 1)) for i in range(3)]
    assert metrics.genotype_diversity(generation(clones)) == 0.0


def test_failure_rate_counts_this_generations_evaluations():
    rows = FAMILY + [individual(5, 1.0, 1, 1, origin="offspring", parents=[2], failed=True),
                     individual(6, 1.0, 1, 1, origin="offspring", parents=[2])]
    assert metrics.failure_rate(generation(rows)) == 0.25


def test_operator_statistics_are_empty_without_offspring():
    samples = [individual(i, 1.0, 1, 1, origin="random") for i in range(3)]
    view = generation(samples)
    for name in ("success_rate", "improvement_over_parent", "survival_rate", "parent_fraction"):
        assert math.isnan(POPULATION_METRICS[name](view)), name


def test_end_to_end_through_the_harness(tmp_path):
    from test_ea import REGISTRY   # random search + the test-only ToyEA

    cfg = ExperimentConfig(
        variants=("random_search", "toy"), seeds=(0,),
        population_size=6, generations=3, offspring_per_generation=6, sim_duration=0.5,
        results_dir=tmp_path / "results", figures_dir=tmp_path / "figures",
    )
    for variant in cfg.variants:
        runner.execute(variant, 0, cfg, REGISTRY)
    tidy = dataset.load(cfg, METRICS, POPULATION_METRICS)

    expected = {"mean_sigma", "best_sigma", "mean_distance", "best_distance", *POPULATION_METRICS}
    assert expected <= set(tidy.columns)

    toy = tidy[tidy.variant == "toy"].set_index("generation")
    rnd = tidy[tidy.variant == "random_search"].set_index("generation")
    later = toy.loc[1:]
    assert later.success_rate.between(0, 1).all()
    assert later.parent_fraction.between(0, 1).all()
    assert later.survival_rate.loc[:2].between(0, 1).all()
    assert math.isnan(toy.survival_rate.loc[3])                 # last generation
    assert (toy.failure_rate == 0).all()
    assert (toy.genotype_diversity > 0).all()
    assert rnd.success_rate.isna().all()                        # no parents
    assert rnd.mean_sigma.isna().all()                          # no σ
    assert np.isfinite(rnd.mean_distance).all()
