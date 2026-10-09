"""Small mutation-strength pilot; refuses to replace existing outputs."""

# ruff: noqa: E402 -- local packages need the assignment paths below.

import argparse
import json
import multiprocessing
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from harness import dataset, runner
from harness.variants import Variant
from neuroevolution.config import ExperimentConfig, outputs
from neuroevolution.ea import StaticSigmaEA
from neuroevolution.metrics import METRICS, POPULATION_METRICS
from neuroevolution.variants import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="New directory name under outputs/")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", choices=("screen", "confirm"), default="screen")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a single directory name")
    if args.workers < 1:
        parser.error("workers must be positive")

    strengths = {"fixed_003": 0.03, "fixed_010": 0.1, "fixed_030": 0.3}
    if args.phase == "confirm":
        strengths = {"fixed_003": 0.03}
    names = ("self_adaptive",) if args.phase == "confirm" else ("random_search", "self_adaptive")
    registry = {name: VARIANTS[name] for name in names}
    registry.update({name: Variant(StaticSigmaEA, f"Fixed sigma {sigma}", "#0072B2")
                     for name, sigma in strengths.items()})
    cfg = ExperimentConfig(
        variants=tuple(registry), seeds=(8101, 8102, 8103),
        population_size=30, offspring_per_generation=30,
        generations=100 if args.phase == "confirm" else 20,
        initial_sigma=0.03 if args.phase == "confirm" else 0.1,
        sim_duration=15.0, workers=args.workers, **outputs(args.name),
    )
    root = cfg.results_dir.parent
    root.mkdir(parents=True, exist_ok=False)
    (root / "pilot_plan.json").write_text(json.dumps({
        "config": cfg.as_dict(), "fixed_strengths": strengths,
        "purpose": "Exploratory pilot, not final results or a significance test",
        "phase": args.phase,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_diff": subprocess.check_output(["git", "diff"], text=True),
        "pilot_script": Path(__file__).read_text(),
    }, indent=2))
    print(f"Pilot outputs: {root}", flush=True)
    jobs = [(variant, seed, replace(cfg, static_sigma=strengths.get(variant, 0.1)))
            for variant in cfg.variants for seed in cfg.seeds]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(runner.execute, variant, seed, settings, registry): (variant, seed)
                   for variant, seed, settings in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            meta = future.result()
            if meta["evaluations"] != cfg.evaluation_budget:
                raise RuntimeError(f"Incorrect budget: {meta}")
            print(f"{index}/{len(jobs)} {meta['variant']} seed={meta['seed']}: "
                  f"{meta['wall_seconds']}s, {meta['evaluations']} evaluations", flush=True)

    tidy = dataset.load(cfg, METRICS, POPULATION_METRICS)
    tidy.to_json(root / "per_generation.json", orient="records", indent=2)
    final = tidy.sort_values("generation").groupby(["variant", "seed"], observed=True).tail(1)
    final.to_json(root / "final_per_seed.json", orient="records", indent=2)
    columns = ["variant", "seed", "best_so_far", "mean_sigma", "genotype_diversity",
               "success_rate", "failure_rate"]
    print(final[columns].to_string(index=False))


if __name__ == "__main__":
    main()
