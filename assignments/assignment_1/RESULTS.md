# Full experiment results

Run on 18 September 2026 using algorithm commit `8cbdd5a`.
These replace the earlier smoke-test outputs.

- Three variants, seeds 0–9: 30 runs total.
- Population 50, 100 generations, 50 offspring per generation.
- 5,050 fitness evaluations per run, including initialization.
- Module cap 20; crossover and mutation rates 0.9; tournament size 3.

[Figures](figures/) and [tables](tables/) are from the full run.
`tables/per_generation.csv` contains the generation-level data;
`tables/final_per_seed.csv` contains the 30 final outcomes.
Lower fitness is better.

The tables and figures use the existing experiment analysis unchanged.
`pairwise_tests.csv` contains its unpaired Mann–Whitney tests, not paired tests.
The raw SQLite databases are kept locally and are not included here (205 MB).

To rerun from the repository root:

```sh
uv run assignments/assignment_1/run_experiment.py
```
