#!/usr/bin/env python
"""Entry point for Standard Assignment 1's experiment.

    python run_experiment.py            # full grid, then tables + figures
    python run_experiment.py --smoke    # same code paths, ~30x less compute

Results land in `results/<variant>/seed_NN/database.db`, figures in `figures/`,
tables in `tables/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# needed to reach the local experimentation code and the shared harness
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from experiment import analysis, figures, variants  # noqa: E402
from experiment.config import SMOKE, ExperimentConfig  # noqa: E402
from harness import dataset, runner  # noqa: E402
from harness.analysis import report  # noqa: E402


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run(cfg: ExperimentConfig) -> None:
    _print_header("RUNNING EXPERIMENT")
    runner.run_all(cfg, variants.VARIANTS)


def analyse(cfg: ExperimentConfig) -> None:
    _print_header("ANALYSIS")

    target_body_facts = analysis.target_set_facts()
    tidy = dataset.load(cfg, analysis.METRICS)
    final, _, _ = report(tidy, cfg)
    print(f"  Fitness lower bound (not an attained optimum): {target_body_facts.fitness_floor:.3f}\n")

    champions = dataset.champion_genotypes(cfg)
    figures.render_all(tidy, final, target_body_facts, champions, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny ExperimentConfig that still executes the same code")
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else ExperimentConfig()

    run(cfg)
    analyse(cfg)


if __name__ == "__main__":
    main()
