"""Statistics for the report: the theoretical floor, summary tables, tests.

Two things here are worth understanding before reading any result.

The floor
---------
No body can be arbitrarily close to all five targets at once, because the
targets differ from each other. Tree edit distance with these unit costs is a
metric, so for any candidate C and any two targets A, B the triangle
inequality gives `d(C,A) + d(C,B) >= d(A,B)`. Summing over all pairs (each
target appears in k-1 of them) bounds the mean distance from below, and since
the standard-deviation term is non-negative, that bound applies to the fitness
itself. It is a *provable* lower bound, not an estimate.

Unit of analysis
----------------
Every test uses one observation per independent run, so N is the number of
seeds. Treating individuals as observations would inflate N by a factor of
thousands and produce meaningless p-values.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .fitness import load_targets, tree_edit_distance


@dataclass(frozen=True)
class TargetSetFacts:
    """Properties of the target set that bound what any EA can achieve."""

    sizes: tuple[int, ...]
    pairwise: np.ndarray
    mean_pairwise: float
    fitness_floor: float
    idealised_best_size: int
    idealised_best_fitness: float

    def summary_lines(self) -> list[str]:
        return [
            f"targets            : {len(self.sizes)} bodies, sizes {list(self.sizes)}",
            f"mean pairwise TED  : {self.mean_pairwise:.2f}",
            f"provable floor     : {self.fitness_floor:.3f}  (triangle inequality)",
            f"idealised optimum  : {self.idealised_best_size} modules "
            f"-> {self.idealised_best_fitness:.2f} (size-only estimate)",
        ]


def _idealised_size_curve(sizes: tuple[int, ...], max_size: int = 40) -> dict[int, float]:
    """Fitness a body of size n would get if it matched perfectly on overlap.

    `|n - m|` is a lower bound on the edit distance to a target of size m, so
    this says where the size-driven optimum sits. The mean term is a genuine
    bound; the std term is indicative only, since std is not monotone in the
    per-target distances.
    """
    curve: dict[int, float] = {}
    for n in range(1, max_size + 1):
        deltas = [abs(n - m) for m in sizes]
        curve[n] = statistics.fmean(deltas) + statistics.pstdev(deltas)
    return curve


def target_set_facts() -> TargetSetFacts:
    """
    Compute statistics about the target bodies alone.
    
    A few interesting findings ones:
    1. Optimal fitness - using traingle inequality we can deduce
        what obsolute lowest fitness value could be achieved in theory.
        technically it's just the middle point between all target bodies

        This is really nice as now we can trully see how good our solution can
        get relative to an optimum.

    2. Optimal solution size - we can exploit the fitness function
        to see which solution size could technically yield the lowest fitness.
        We do this by assuming that the solution and target bodies math in the
        module types, but only differ in module counts (i.e. relabeling is free)

        Following this, transforming solution of size `n` into target mody of
        size `m` needs |n-m| operations (and because each node addition and
        deletion) costs exactly 1, we can calculate idealized fitness for each `n`.
    """
    targets = load_targets()
    count = len(targets)
    sizes = tuple(target.number_of_nodes() for target in targets)

    target_pairwise_distances = np.zeros((count, count))
    for i, j in itertools.combinations(range(count), 2):
        distance = tree_edit_distance(targets[i], targets[j])
        target_pairwise_distances[i, j] = target_pairwise_distances[j, i] = distance

    upper = [target_pairwise_distances[i, j] for i, j in itertools.combinations(range(count), 2)]
    mean_pairwise = statistics.fmean(upper)

    # (k-1) * sum_A d(C,A) >= sum_pairs d(A,B)   =>   mean >= S / (k(k-1))
    floor = sum(upper) / (count * (count - 1))

    curve = _idealised_size_curve(sizes)
    best_size = min(curve, key=lambda n: curve[n])

    return TargetSetFacts(
        sizes=sizes,
        pairwise=target_pairwise_distances,
        mean_pairwise=mean_pairwise,
        fitness_floor=floor,
        idealised_best_size=best_size,
        idealised_best_fitness=curve[best_size],
    )


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
    sizes = final.groupby("variant", observed=True)["mean_size"].mean()
    table["mean_final_size"] = sizes
    return table.round(3)

def pairwise_tests(final: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U on end-of-run fitness for every pair of variants.

    Non-parametric because run outcomes are not reliably normal and N is
    small.
    """
    variants = sorted(final["variant"].unique())

    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(variants, 2):
        a = final.loc[final["variant"] == left, "best_so_far"].to_numpy()
        b = final.loc[final["variant"] == right, "best_so_far"].to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "variant_a": left,
                "variant_b": right,
                "n_a": len(a),
                "n_b": len(b),
                "median_a": round(float(np.median(a)), 3),
                "median_b": round(float(np.median(b)), 3),
                "U": float(result.statistic),
                "p": float(result.pvalue),
                "significant_0.05": bool(result.pvalue < 0.05),
            }
        )
    return pd.DataFrame(rows)
