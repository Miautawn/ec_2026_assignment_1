"""Every figure that goes into the report, saved as a PNG.

Conventions kept deliberately uniform so the figures read as one set:
one colour per variant (colour-blind-safe), the provable fitness floor drawn
on every fitness axis, and error bands that are always "spread across
independent runs" -- never the standard-deviation term inside the fitness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import variants
from .analysis import TargetSetFacts
from .config import ExperimentConfig
from .fitness import per_target_distances

FIGSIZE = (6.0, 3.6)
DPI = 300
GRID_STYLE = {"alpha": 0.3, "linewidth": 0.5}


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(**GRID_STYLE)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_floor(ax: plt.Axes, facts: TargetSetFacts) -> None:
    """Mark the provable lower bound on fitness."""
    ax.axhline(
        facts.fitness_floor,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"provable floor ({facts.fitness_floor:.2f})",
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _band(
    ax: plt.Axes,
    tidy: pd.DataFrame,
    column: str,
    variant: str,
) -> None:
    """Mean across seeds, with a +/-1 std band, for one variant."""
    subset = tidy.loc[tidy["variant"] == variant]
    grouped = subset.groupby("generation", observed=True)[column]
    mean, std = grouped.mean(), grouped.std(ddof=1).fillna(0.0)
    generations = mean.index.to_numpy()
    colour = variants.colour(variant)

    ax.plot(generations, mean.to_numpy(), color=colour, linewidth=1.8,
            label=variants.label(variant))
    ax.fill_between(
        generations,
        (mean - std).to_numpy(),
        (mean + std).to_numpy(),
        color=colour,
        alpha=0.18,
        linewidth=0,
    )


# --------------------------------------------------------------------------- #
#  Figure 1 -- the one the brief explicitly requires
# --------------------------------------------------------------------------- #

def convergence(tidy: pd.DataFrame, facts: TargetSetFacts, out: Path) -> Path:
    """Best-so-far fitness across generations, mean +/- std over runs.

    This is the assignment's mandated line plot. Best-so-far (rather than
    best-in-generation) is used because it is the only curve that compares
    fairly against random search, which has no population continuity.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for variant in tidy["variant"].cat.categories:
        _band(ax, tidy, "best_so_far", variant)
    _draw_floor(ax, facts)
    n_seeds = tidy.groupby("variant", observed=True)["seed"].nunique().max()
    _style_axes(
        ax,
        "Generation",
        "Fitness (lower is better)",
        f"Convergence: best-so-far, mean $\\pm$ 1 s.d. over {n_seeds} independent runs",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, out / "fig1_convergence.png")


# --------------------------------------------------------------------------- #
#  Figure 3 -- final best fitness distribution in a box-plot
# --------------------------------------------------------------------------- #

def final_distribution(final: pd.DataFrame, facts: TargetSetFacts, out: Path) -> Path:
    """Box plot of end-of-run fitness with every individual run overlaid.

    Overlaying the raw points matters at these sample sizes: a box plot of ten
    numbers can hide the fact that one run behaved completely differently.
    """
    order = list(final["variant"].cat.categories)
    data = [final.loc[final["variant"] == v, "best_so_far"].to_numpy() for v in order]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    boxes = ax.boxplot(
        data, patch_artist=True, widths=0.55, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, variant in zip(boxes["boxes"], order, strict=True):
        patch.set_facecolor(variants.colour(variant))
        patch.set_alpha(0.35)
        patch.set_edgecolor(variants.colour(variant))

    rng = np.random.default_rng(0)  # jitter only; not part of any result
    for index, (values, variant) in enumerate(zip(data, order, strict=True), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(index + jitter, values, s=18, zorder=3,
                   color=variants.colour(variant), edgecolor="white", linewidth=0.5)

    _draw_floor(ax, facts)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(
        [variants.label(v).replace(" (", "\n(") for v in order], fontsize=8
    )
    _style_axes(ax, "", "Final best fitness (lower is better)",
                "End-of-run outcome, one point per independent run")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out / "fig3_final_distribution.png")


# --------------------------------------------------------------------------- #
#  Figure 4 -- bloat
# --------------------------------------------------------------------------- #

def genome_size(tidy: pd.DataFrame, facts: TargetSetFacts, out: Path) -> Path:
    """Mean module count per generation, against the idealised optimum.
    """
    if tidy["mean_size"].isna().all():
        print("  skipped fig4 (no module counts -- non-tree encoding?)")
        return out / "fig4_genome_size.png"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for variant in tidy["variant"].cat.categories:
        _band(ax, tidy, "mean_size", variant)
    ax.axhline(
        facts.idealised_best_size, color="black", linestyle=":", linewidth=1.0,
        label=f"idealised optimum ({facts.idealised_best_size} modules)",
    )
    _style_axes(ax, "Generation", "Mean modules per body",
                "Genome size over time (bloat)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, out / "fig4_genome_size.png")


# --------------------------------------------------------------------------- #
#  Figure 5 -- Heatmap of pairwise distances between target bodies
#   and idealized fitness curve for different solution bloat
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
    _style_axes(ax, "Body size (modules)", "Idealised fitness (lower is better)",
                "Where the fitness metric pulls body size")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    return _save(fig, out / "fig5_target_set.png")


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
            color=variants.colour(variant), alpha=0.85,
            label=f"{variants.label(variant)} (fit. {champions[variant]['fitness']:.2f})",
        )
    ax.set_xticks(x, [f"target_{i:02d}" for i in range(n_targets)], fontsize=8)
    _style_axes(ax, "", "Tree edit distance",
                "Per-target distance of each variant's best body")
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, out / "fig6_champion_breakdown.png")


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
    return [
        convergence(tidy, facts, out),
        final_distribution(final, facts, out),
        genome_size(tidy, facts, out),
        target_set(facts, out),
        champion_breakdown(champions, out),
    ]
