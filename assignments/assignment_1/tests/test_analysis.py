"""Paired analysis and separation of smoke outputs from final results."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.analysis import pairwise_tests
from experiment.config import ExperimentConfig, SMOKE


def outcomes():
    return pd.DataFrame([
        {"variant": name, "seed": seed, "best_so_far": value}
        for name, values in {"a": [0] * 10, "b": [1] * 6 + [-1] * 4,
                             "c": [2] * 10}.items()
        for seed, value in enumerate(values)
    ])


def test_exact_sign_tests_and_holm():
    data = outcomes()
    result = pairwise_tests(data)
    assert result.p.tolist() == pytest.approx([0.75390625, 0.001953125, 0.001953125])
    assert result.p_holm.tolist() == pytest.approx([0.75390625, 0.005859375, 0.005859375])
    assert result.wins_a.tolist() == [6, 10, 10]
    pd.testing.assert_frame_equal(result, pairwise_tests(data.sample(frac=1, random_state=4)))


def test_ties():
    data = outcomes()
    data["best_so_far"] = 0
    result = pairwise_tests(data)
    assert result.ties.tolist() == [10, 10, 10]
    assert result.p.tolist() == [1, 1, 1]
    data.loc[(data.variant == "a") & (data.seed == 0), "best_so_far"] = -1
    assert pairwise_tests(data).iloc[0].ties == 9


def test_invalid_pairing():
    data = outcomes()
    with pytest.raises(ValueError, match="Seed sets differ"):
        pairwise_tests(data.iloc[1:])
    with pytest.raises(ValueError, match="one final outcome"):
        pairwise_tests(pd.concat([data, data.iloc[:1]]))
    data.loc[0, "best_so_far"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pairwise_tests(data)


def test_smoke_does_not_overwrite_final_outputs():
    full = ExperimentConfig()
    assert SMOKE.results_dir.parent != full.results_dir.parent
    assert SMOKE.figures_dir != full.figures_dir
