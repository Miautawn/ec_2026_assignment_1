#!/usr/bin/env python
"""Entry point for Standard Assignment 1's experiment.

    python run_experiment.py targets          # bounds from the target set alone
    python run_experiment.py run --smoke      # fast end-to-end check (~1 min)
    python run_experiment.py all              # full grid, then tables + figures
    python run_experiment.py analyse          # re-analyse without re-running

Results land in `results/<variant>/seed_NN/database.db`, figures in `figures/`,
tables in `tables/`. """

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# needed to reach the local experimentaiton utils
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import analysis, dataset, figures, runner  # noqa: E402
from experiment.config import SMOKE, ExperimentConfig  # noqa: E402


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run(cfg: ExperimentConfig) -> None:
    _print_header("RUNNING EXPERIMENT")
    print(f"  variants  : {list(cfg.variants)}")
    print(f"  seeds     : {list(cfg.seeds)}  ({len(cfg.seeds)} independent runs each)")
    print(f"  budget    : {cfg.evaluation_budget} evaluations per run "
          f"(pop {cfg.population_size} + {cfg.generations} gen x "
          f"{cfg.offspring_per_generation} offspring)")
    print(f"  results   : {cfg.results_dir}\n")
    runner.run_all(cfg)


def analyse(cfg: ExperimentConfig) -> None:
    _print_header("ANALYSIS")

    target_body_facts = analysis.target_set_facts()
    tidy = dataset.load(cfg)
    final = dataset.final_per_seed(tidy)

    tables_dir = cfg.results_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tidy.to_csv(tables_dir / "per_generation.csv", index=False)
    final.to_csv(tables_dir / "final_per_seed.csv", index=False)

    summary = analysis.summary_table(final)
    tests = analysis.pairwise_tests(final)
    summary.to_csv(tables_dir / "summary.csv")
    tests.to_csv(tables_dir / "pairwise_tests.csv", index=False)

    print("\nFinal best fitness per variant (across independent runs):")
    print(summary.to_string())
    print(f"\n  Best possible fitness: {target_body_facts.fitness_floor:.3f}")

    print("\nPairwise Mann-Whitney U on final best fitness:")
    print(tests.to_string(index=False) if not tests.empty else "  (nothing to compare)")

    print()
    champions = dataset.champion_genotypes(cfg)
    figures.render_all(tidy, final, target_body_facts, champions, cfg)
    print(f"\n  tables written to {tables_dir}")



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
