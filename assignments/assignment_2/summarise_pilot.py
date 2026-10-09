"""Summarise completed pilot runs without treating them as final experiments."""

# ruff: noqa: E402 -- configure plotting and local imports before loading packages.

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from harness import dataset
from neuroevolution.config import ExperimentConfig
from neuroevolution.metrics import METRICS, POPULATION_METRICS


def summarise(root):
    settings = json.loads((root / "pilot_plan.json").read_text())["config"]
    settings.pop("evaluation_budget")
    for name in ("results_dir", "figures_dir"):
        settings[name] = Path(settings[name])
    cfg = ExperimentConfig(**settings)
    runs = dataset.discover_runs(cfg)
    expected = {(v, s) for v in cfg.variants for s in cfg.seeds}
    if {(v, s) for _, v, s in runs} != expected:
        raise ValueError("Pilot is incomplete")
    tidy = dataset.load(cfg, METRICS, POPULATION_METRICS)
    results = []
    for db, variant, seed in runs:
        with sqlite3.connect(db) as conn:
            tags = [json.loads(row[0]) for row in conn.execute("SELECT tags_ FROM individual")]
        if len(tags) != cfg.evaluation_budget:
            raise ValueError(f"Incorrect evaluation budget: {variant}, {seed}")
        rows = tidy[(tidy.variant == variant) & (tidy.seed == seed)].sort_values("generation")
        first, final = rows.iloc[0], rows.iloc[-1]
        cutoff = int(cfg.generations * 0.75)
        earlier = rows[rows.generation == cutoff].iloc[0]
        sigma = None if np.isnan(final.mean_sigma) else float(final.mean_sigma)
        results.append({
            "variant": variant, "seed": seed,
            "evaluations": len(tags), "initial_best": float(first.best_so_far),
            "final_best": float(final.best_so_far),
            "improvement_last_quarter": float(earlier.best_so_far - final.best_so_far),
            "final_survivor_mean_sigma": sigma,
            "final_genotype_diversity": float(final.genotype_diversity),
            "physics_failures": sum(bool(t.get("failed")) for t in tags),
            "sigma_bound_hits": sum(bool(t.get("sigma_at_bound")) for t in tags),
        })
    summary = {v: float(np.mean([r["final_best"] for r in results if r["variant"] == v]))
               for v in cfg.variants}
    (root / "diagnostics.json").write_text(json.dumps({
        "status": "pilot only", "mean_final_best": summary, "runs": results,
    }, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    for index, (variant, group) in enumerate(tidy.groupby("variant", observed=True)):
        grouped = group.groupby("generation")
        x = grouped.evaluations_so_far.mean()
        for ax, column in zip(axes, ("best_so_far", "mean_sigma", "genotype_diversity")):
            y = grouped[column].mean()
            if y.notna().any():
                line, = ax.plot(x, y, label=variant, color=f"C{index}")
                sd = grouped[column].std().fillna(0)
                ax.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=0.15)
    for ax, label in zip(axes, ("Best-so-far distance (m)", "Survivor mean sigma", "Weight diversity")):
        ax.set(xlabel="Evaluations", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pilot runs: mean and one standard deviation across seeds")
    fig.savefig(root / "diagnostics.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"mean_final_best": summary, "runs": results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarise(parser.parse_args().directory)
