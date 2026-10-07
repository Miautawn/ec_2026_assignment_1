"""Statistics for the report: a summary per EA, and every pair of EAs compared.

Each run counts once (N = number of seeds, never individuals), and runs are
compared seed by seed. The main result of a comparison is a confidence
interval for the difference: see `pairwise_tests`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import BaseConfig
from .dataset import final_per_seed, metric_names


def summary_table(final: pd.DataFrame) -> pd.DataFrame:
    """Per-variant summary of end-of-run outcomes, across seeds."""
    grouped = final.groupby("variant", observed=True)["best_so_far"]
    table = pd.DataFrame(
        {
            "runs": grouped.count(),
            "mean": grouped.mean(),
            "std": grouped.std(ddof=1),
            "min": grouped.min(),
            "median": grouped.median(),
            "max": grouped.max(),
        }
    )
    for name in metric_names(final):
        table[f"mean_final_{name}"] = (
            final.groupby("variant", observed=True)[f"mean_{name}"].mean()
        )
    return table.round(3)


#: Bootstrap settings. The seed makes every confidence interval reproducible.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CONFIDENCE = 0.95


def _holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def _sign_test(differences: np.ndarray) -> float:
    """Exact two-sided sign test; ties carry no information and are dropped."""
    lower = int(np.count_nonzero(differences < 0))
    non_ties = lower + int(np.count_nonzero(differences > 0))
    return float(stats.binomtest(lower, non_ties, p=0.5).pvalue) if non_ties else 1.0


def _wilcoxon(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test; zero differences are dropped."""
    if not np.any(differences):
        return 1.0
    return float(stats.wilcoxon(differences, zero_method="wilcox").pvalue)


def mean_difference_ci(differences: np.ndarray) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean paired difference.

    BCa (bias-corrected and accelerated), which adjusts for skew -- worth it
    with only ~10 seeds. Falls back to the percentile method if BCa is
    undefined, and is a single point when every difference is identical.
    """
    if len(differences) < 2:
        return (float("nan"), float("nan"))
    if np.ptp(differences) == 0:
        return (float(differences[0]), float(differences[0]))
    for method in ("BCa", "percentile"):
        interval = stats.bootstrap(
            (differences,),
            np.mean,
            n_resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=CONFIDENCE,
            method=method,
            rng=np.random.default_rng(BOOTSTRAP_SEED),
        ).confidence_interval
        if np.isfinite([interval.low, interval.high]).all():
            return (float(interval.low), float(interval.high))
    return (float("nan"), float("nan"))


def pairwise_tests(final: pd.DataFrame) -> pd.DataFrame:
    """Compare every pair of EAs (A, B) on final best fitness, seed by seed.

    Reports how often each was lower, the mean difference A - B with a 95%
    bootstrap confidence interval, and two Holm-corrected tests: the sign test
    (direction only) and Wilcoxon (also uses the size of the differences).

    Reading the interval: excludes 0, there is a difference of roughly this size;
    narrow around 0, any difference is too small to matter; wide around 0,
    inconclusive, and more seeds would narrow it.
    """
    if final.duplicated(["variant", "seed"]).any():
        raise ValueError("Expected one final outcome per variant and seed")
    if not np.isfinite(final["best_so_far"].to_numpy()).all():
        raise ValueError("Final fitness must be finite")
    variants = sorted(final["variant"].unique())

    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(variants, 2):
        a = final.loc[final["variant"] == left].set_index("seed")["best_so_far"].sort_index()
        b = final.loc[final["variant"] == right].set_index("seed")["best_so_far"].sort_index()
        if not a.index.equals(b.index):
            raise ValueError(f"Seed sets differ for {left} and {right}")
        differences = (a - b).to_numpy()
        a_lower = int(np.count_nonzero(differences < 0))
        b_lower = int(np.count_nonzero(differences > 0))
        ci_low, ci_high = mean_difference_ci(differences)
        rows.append(
            {
                "variant_a": left,
                "variant_b": right,
                "pairs": len(a),
                "a_lower": a_lower,
                "b_lower": b_lower,
                "ties": len(a) - a_lower - b_lower,
                "mean_difference_a_minus_b": float(differences.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "median_a": round(float(np.median(a)), 3),
                "median_b": round(float(np.median(b)), 3),
                "p_sign": _sign_test(differences),
                "p_wilcoxon": _wilcoxon(differences),
            }
        )

    for test in ("p_sign", "p_wilcoxon"):
        for row, adjusted in zip(rows, _holm([row[test] for row in rows]), strict=True):
            row[f"{test}_holm"] = adjusted
    return pd.DataFrame(rows)


def report(
    tidy: pd.DataFrame, cfg: BaseConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the end-of-run tables, write all four CSVs, and print them.

    Returns `(final, summary, tests)` for the caller's figures.
    """
    final = final_per_seed(tidy)
    summary = summary_table(final)
    tests = pairwise_tests(final)

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(cfg.tables_dir / "per_generation.csv", index=False)
    final.to_csv(cfg.tables_dir / "final_per_seed.csv", index=False)
    summary.to_csv(cfg.tables_dir / "summary.csv")
    tests.to_csv(cfg.tables_dir / "pairwise_tests.csv", index=False)

    print("\nFinal best fitness per variant (across independent runs):")
    print(summary.to_string())
    print("\nPaired comparisons on final best fitness "
          "(mean difference A - B with 95% bootstrap CI; Holm-corrected tests):")
    print(tests.to_string(index=False) if not tests.empty else "  (nothing to compare)")
    print(f"\n  tables written to {cfg.tables_dir}")
    return final, summary, tests
