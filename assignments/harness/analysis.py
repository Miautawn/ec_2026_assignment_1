"""Statistics for the report: summary table and paired significance tests.

Unit of analysis
----------------
Every test uses one observation per independent run, so N is the number of
seeds. Treating individuals as observations would inflate N by a factor of
thousands and produce meaningless p-values. Runs are paired by seed: variants
sharing a seed start from the same initial population.
"""

from __future__ import annotations

import itertools

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


def pairwise_tests(final: pd.DataFrame) -> pd.DataFrame:
    """Exact paired sign tests by seed, with Holm correction across pairs."""
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
        wins = int(np.count_nonzero(differences < 0))
        losses = int(np.count_nonzero(differences > 0))
        non_ties = wins + losses
        p = stats.binomtest(wins, non_ties, p=0.5).pvalue if non_ties else 1.0
        rows.append(
            {
                "variant_a": left,
                "variant_b": right,
                "pairs": len(a),
                "wins_a": wins,
                "wins_b": losses,
                "ties": len(a) - non_ties,
                "mean_difference_a_minus_b": float(differences.mean()),
                "median_a": round(float(np.median(a)), 3),
                "median_b": round(float(np.median(b)), 3),
                "p": float(p),
            }
        )
    adjusted = 0.0
    for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]["p"])):
        adjusted = max(adjusted, min(1.0, (len(rows) - rank) * rows[index]["p"]))
        rows[index]["p_holm"] = adjusted
        rows[index]["significant_0.05"] = adjusted < 0.05
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
    print("\nPaired sign tests on final best fitness (Holm correction):")
    print(tests.to_string(index=False) if not tests.empty else "  (nothing to compare)")
    print(f"\n  tables written to {cfg.tables_dir}")
    return final, summary, tests
