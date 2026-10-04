"""Plotting primitives and the figures every assignment needs, saved as PNGs.

Conventions kept deliberately uniform so the figures read as one set: one
colour per variant (taken from the variant registry), an optional reference
line on fitness axes, and error bands that are always "spread across
independent runs" -- never a spread term inside the fitness itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .variants import Variant

FIGSIZE = (6.0, 3.6)
DPI = 300
GRID_STYLE = {"alpha": 0.3, "linewidth": 0.5}


class Reference(NamedTuple):
    """A horizontal reference line, e.g. a provable fitness floor."""

    value: float
    label: str


# --------------------------------------------------------------------------- #
#  Primitives
# --------------------------------------------------------------------------- #

def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(**GRID_STYLE)
    ax.spines[["top", "right"]].set_visible(False)


def draw_reference(ax: plt.Axes, reference: Reference) -> None:
    ax.axhline(
        reference.value,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=reference.label,
    )


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def band(
    ax: plt.Axes,
    tidy: pd.DataFrame,
    column: str,
    variant: str,
    registry: Mapping[str, Variant],
) -> None:
    """Mean across seeds, with a +/-1 std band, for one variant."""
    subset = tidy.loc[tidy["variant"] == variant]
    grouped = subset.groupby("generation", observed=True)[column]
    mean, std = grouped.mean(), grouped.std(ddof=1).fillna(0.0)
    generations = mean.index.to_numpy()
    colour = registry[variant].colour

    ax.plot(generations, mean.to_numpy(), color=colour, linewidth=1.8,
            label=registry[variant].label)
    ax.fill_between(
        generations,
        (mean - std).to_numpy(),
        (mean + std).to_numpy(),
        color=colour,
        alpha=0.18,
        linewidth=0,
    )


# --------------------------------------------------------------------------- #
#  Shared figures
# --------------------------------------------------------------------------- #

def over_generations(
    tidy: pd.DataFrame,
    column: str,
    registry: Mapping[str, Variant],
    path: Path,
    *,
    ylabel: str,
    title: str,
    reference: Reference | None = None,
) -> Path:
    """Any per-generation column, mean +/- std over runs, one band per variant."""
    if tidy[column].isna().all():
        print(f"  skipped {path.name} (no data in {column})")
        return path

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for variant in tidy["variant"].cat.categories:
        band(ax, tidy, column, variant, registry)
    if reference is not None:
        draw_reference(ax, reference)
    style_axes(ax, "Generation", ylabel, title)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return save(fig, path)


def convergence(
    tidy: pd.DataFrame,
    registry: Mapping[str, Variant],
    path: Path,
    reference: Reference | None = None,
) -> Path:
    """Best-so-far fitness across generations, mean +/- std over runs.

    The briefs' mandated line plot. Best-so-far (rather than
    best-in-generation) is used because it is the only curve that compares
    fairly against random search, which has no population continuity.
    """
    n_seeds = tidy.groupby("variant", observed=True)["seed"].nunique().max()
    return over_generations(
        tidy,
        "best_so_far",
        registry,
        path,
        ylabel="Fitness (lower is better)",
        title=f"Convergence: best-so-far, mean $\\pm$ 1 s.d. over {n_seeds} independent runs",
        reference=reference,
    )


def final_distribution(
    final: pd.DataFrame,
    registry: Mapping[str, Variant],
    path: Path,
    reference: Reference | None = None,
) -> Path:
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
        patch.set_facecolor(registry[variant].colour)
        patch.set_alpha(0.35)
        patch.set_edgecolor(registry[variant].colour)

    rng = np.random.default_rng(0)  # jitter only; not part of any result
    for index, (values, variant) in enumerate(zip(data, order, strict=True), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(index + jitter, values, s=18, zorder=3,
                   color=registry[variant].colour, edgecolor="white", linewidth=0.5)

    if reference is not None:
        draw_reference(ax, reference)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(
        [registry[v].label.replace(" (", "\n(") for v in order], fontsize=8
    )
    style_axes(ax, "", "Final best fitness (lower is better)",
               "End-of-run outcome, one point per independent run")
    if reference is not None:
        ax.legend(frameon=False, fontsize=8)
    return save(fig, path)
