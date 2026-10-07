"""Assignment 2's own figure: each EA's best robot path, drawn over the terrain.

The other figures are shared, in `harness.figures`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import matplotlib.pyplot as plt
import mujoco as mj

from harness.figures import save, style_axes
from harness.variants import Variant

from .config import ExperimentConfig
from .evaluator import Evaluator
from .replay import Champion


def trajectories(
    cfg: ExperimentConfig,
    champions: Mapping[str, Champion],
    registry: Mapping[str, Variant],
    path: Path,
) -> Path:
    """Top-down paths of each variant's champion over the terrain."""
    evaluator = Evaluator(cfg)
    model = evaluator.scene.model

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    hfields = [g for g in range(model.ngeom) if model.geom_type[g] == mj.mjtGeom.mjGEOM_HFIELD]
    if hfields:
        hfield = model.geom_dataid[hfields[0]]
        rows, cols = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
        half_x, half_y, top, _ = model.hfield_size[hfield]
        centre = model.geom_pos[hfields[0]]
        heights = model.hfield_data[model.hfield_adr[hfield] : model.hfield_adr[hfield] + rows * cols]
        image = ax.imshow(
            heights.reshape(rows, cols) * top, origin="lower", cmap="Greys",
            extent=(centre[0] - half_x, centre[0] + half_x, centre[1] - half_y, centre[1] + half_y),
        )
        fig.colorbar(image, ax=ax, fraction=0.046, label="terrain height (m)")

    spawn, target = evaluator.scene.spawn_xy, evaluator.scene.target_xy
    for variant, champion in champions.items():
        result = evaluator.evaluate(champion.weights, record_trajectory=True)
        xy = result.trajectory[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=registry[variant].colour, linewidth=1.8,
                label=f"{registry[variant].label} (fit. {champion.fitness:.2f})")
        ax.plot(*result.final_xy, "x", color=registry[variant].colour, markersize=8)
    ax.plot(*spawn, "o", color="black", markersize=6, label="start")
    ax.plot(*target, "*", color="#D55E00", markersize=14, label="target")

    reach = max(abs(target[1] - spawn[1]), 1.0) * 1.4
    ax.set_xlim(spawn[0] - reach, spawn[0] + reach)
    ax.set_ylim(min(spawn[1], target[1]) - reach * 0.4, max(spawn[1], target[1]) + reach * 0.4)
    ax.set_aspect("equal")
    style_axes(ax, "x (m)", "y (m)", "Champion paths, seen from above")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    return save(fig, path)
