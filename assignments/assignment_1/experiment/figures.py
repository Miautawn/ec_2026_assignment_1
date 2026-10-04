"""Assignment 1 figures, and the order every figure is rendered in.

The convergence plot, the end-of-run distribution and the per-generation band
plot are shared (`harness.figures`); only figures that need the tree-edit
distance or the target set live here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harness import figures as shared
from harness.figures import FIGSIZE, Reference, save, style_axes

from .analysis import TargetSetFacts
from .config import ExperimentConfig
from .fitness import per_target_distances
from .variants import VARIANTS


# --------------------------------------------------------------------------- #
#  Figure 4 -- bloat
# --------------------------------------------------------------------------- #

def genome_size(tidy: pd.DataFrame, facts: TargetSetFacts, out: Path) -> Path:
    """Mean module count per generation, against the idealised optimum."""
    return shared.over_generations(
        tidy,
        "mean_size",
        VARIANTS,
        out / "fig4_genome_size.png",
        ylabel="Mean modules per body",
        title="Genome size over time (bloat)",
        reference=Reference(
            facts.idealised_best_size,
            f"idealised optimum ({facts.idealised_best_size} modules)",
        ),
    )


# --------------------------------------------------------------------------- #
#  Figure 5 -- idealized fitness curve for different solution bloat
# --------------------------------------------------------------------------- #

def target_set(facts: TargetSetFacts, out: Path) -> Path:
    """Idealised size-only fitness against body size.

    Editing a body of n modules into a target of m costs at least |n - m|, so
    feeding those bounds through the real fitness formula shows which body
    size the metric pulls towards. The mean term is a genuine lower bound; the
    std term is indicative, so the minimum is an estimate, not a proof. The
    provable floor (triangle inequality over the target set) is drawn for
    reference.
    """
    sizes = np.arange(1, 41)
    curve = [
        np.mean([abs(n - m) for m in facts.sizes])
        + np.std([abs(n - m) for m in facts.sizes])
        for n in sizes
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sizes, curve, color="#0072B2", linewidth=1.8,
            label="idealised size-only fitness")
    ax.axvline(
        facts.idealised_best_size, color="black", linestyle=":", linewidth=1.0,
        label=f"estimated optimum: {facts.idealised_best_size} modules",
    )
    ax.axhline(
        facts.fitness_floor, color="#D55E00", linestyle="--", linewidth=1.0,
        label=f"provable floor: {facts.fitness_floor:.2f}",
    )
    for size in facts.sizes:
        ax.axvline(size, color="grey", alpha=0.35, linewidth=0.8)
    ax.text(
        0.98, 0.04,
        f"grey lines: target sizes {list(facts.sizes)}\n"
        f"mean pairwise TED between targets: {facts.mean_pairwise:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#444444",
    )
    style_axes(ax, "Body size (modules)", "Idealised fitness (lower is better)",
                "Where the fitness metric pulls body size")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    return save(fig, out / "fig5_target_set.png")


# --------------------------------------------------------------------------- #
#  Figure 6 -- what the champion actually sacrificed
# --------------------------------------------------------------------------- #

def champion_breakdown(champions: dict, out: Path) -> Path:
    """Per-target distance for each variant's best body.

    The aggregated fitness hides which target a compromise body gives up on;
    """
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome

    rows = {}
    for variant, record in champions.items():
        genotype = record["genotype"]
        if isinstance(genotype, dict) and "nodes" in genotype:
            body = TreeGenome.from_dict(genotype).to_networkx()
            rows[variant] = per_target_distances(body)

    if not rows:
        print("  skipped fig6 (no tree-genome champions found)")
        return out / "fig6_champion_breakdown.png"

    n_targets = len(next(iter(rows.values())))
    x = np.arange(n_targets)
    width = 0.8 / len(rows)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, (variant, distances) in enumerate(rows.items()):
        ax.bar(
            x + index * width - 0.4 + width / 2, distances, width,
            color=VARIANTS[variant].colour, alpha=0.85,
            label=f"{VARIANTS[variant].label} (fit. {champions[variant]['fitness']:.2f})",
        )
    ax.set_xticks(x, [f"target_{i:02d}" for i in range(n_targets)], fontsize=8)
    style_axes(ax, "", "Tree edit distance",
                "Per-target distance of each variant's best body")
    ax.legend(frameon=False, fontsize=7)
    return save(fig, out / "fig6_champion_breakdown.png")


# --------------------------------------------------------------------------- #

def render_all(
    tidy: pd.DataFrame,
    final: pd.DataFrame,
    facts: TargetSetFacts,
    champions: dict,
    cfg: ExperimentConfig,
) -> list[Path]:
    """Render every figure into `cfg.figures_dir`.

    Five figures: the mandated convergence plot, the end-of-run spread,
    bloat, one Methods figure justifying the floor line, and the champion's
    per-target breakdown. A 3-page report will not fit all five -- pick.
    """
    out = cfg.figures_dir
    print(f"rendering figures into {out}")
    floor = Reference(facts.fitness_floor, f"provable floor ({facts.fitness_floor:.2f})")
    return [
        shared.convergence(tidy, VARIANTS, out / "fig1_convergence.png", floor),
        shared.final_distribution(final, VARIANTS, out / "fig3_final_distribution.png", floor),
        genome_size(tidy, facts, out),
        target_set(facts, out),
        champion_breakdown(champions, out),
    ]
