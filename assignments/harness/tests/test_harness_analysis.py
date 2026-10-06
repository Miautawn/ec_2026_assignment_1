"""Paired comparisons and the per-variant summary."""

import numpy as np
import pandas as pd
import pytest

from harness.analysis import mean_difference_ci, pairwise_tests, summary_table


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
    assert result.p_sign.tolist() == pytest.approx([0.75390625, 0.001953125, 0.001953125])
    assert result.p_sign_holm.tolist() == pytest.approx([0.75390625, 0.005859375, 0.005859375])
    assert result.a_lower.tolist() == [6, 10, 10]
    pd.testing.assert_frame_equal(result, pairwise_tests(data.sample(frac=1, random_state=4)))


def test_ties():
    data = outcomes()
    data["best_so_far"] = 0
    result = pairwise_tests(data)
    assert result.ties.tolist() == [10, 10, 10]
    assert result.p_sign.tolist() == [1, 1, 1]
    assert result.p_wilcoxon.tolist() == [1, 1, 1]
    data.loc[(data.variant == "a") & (data.seed == 0), "best_so_far"] = -1
    assert pairwise_tests(data).iloc[0].ties == 9


def test_wilcoxon_uses_the_size_of_differences():
    """Six large losses and four tiny wins: direction alone says nothing."""
    data = pd.DataFrame(
        [{"variant": "a", "seed": s, "best_so_far": 0.0} for s in range(10)]
        + [{"variant": "b", "seed": s, "best_so_far": 10.0 if s < 6 else -0.1}
           for s in range(10)]
    )
    row = pairwise_tests(data).iloc[0]
    assert row.p_sign == pytest.approx(0.75390625)
    assert row.p_wilcoxon == pytest.approx(0.08984375)
    assert row.p_wilcoxon < row.p_sign


def test_confidence_interval_brackets_the_mean_and_is_reproducible():
    differences = np.array([0.3, -0.1, 0.5, 0.2, 0.0, 0.4, -0.2, 0.6, 0.1, 0.3])
    low, high = mean_difference_ci(differences)
    assert low < differences.mean() < high
    assert (low, high) == mean_difference_ci(differences)        # fixed bootstrap seed
    wide = mean_difference_ci(differences * 10)
    assert wide[1] - wide[0] > (high - low) * 5                   # more spread, wider


def test_confidence_interval_edge_cases():
    assert mean_difference_ci(np.array([0.2] * 5)) == (0.2, 0.2)  # no spread: a point
    assert all(np.isnan(mean_difference_ci(np.array([0.2]))))    # one pair: undefined


def test_comparison_table_reports_effect_size():
    row = pairwise_tests(outcomes()).set_index(["variant_a", "variant_b"]).loc[("a", "c")]
    assert row.mean_difference_a_minus_b == -2.0
    assert (row.ci95_low, row.ci95_high) == (-2.0, -2.0)


def test_invalid_pairing():
    data = outcomes()
    with pytest.raises(ValueError, match="Seed sets differ"):
        pairwise_tests(data.iloc[1:])
    with pytest.raises(ValueError, match="one final outcome"):
        pairwise_tests(pd.concat([data, data.iloc[:1]]))
    data.loc[0, "best_so_far"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pairwise_tests(data)


def test_summary_adds_one_column_per_metric():
    data = outcomes().assign(mean_sigma=0.5, mean_fitness=1.0)
    table = summary_table(data)
    assert "mean_final_sigma" in table.columns
    assert "mean_final_fitness" not in table.columns
