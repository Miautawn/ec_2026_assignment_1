# Algorithm notes

Code: `experiment/variants.py`. Settings: `experiment/config.py`.
All variants use the existing runner and assignment fitness (lower is better).

The two EAs differ only in mutation order:

- `MutateChildEA`: select parents → crossover → mutate children → evaluate.
- `MutateParentEA`: select parents → mutate copies → crossover → evaluate.

Original parents stay unchanged. Mutated parent copies aren't evaluated separately.
Random search draws fresh trees; its best-so-far archive doesn't influence sampling.

## Defaults

- 50 initial trees; requested sizes drawn uniformly from 1–20 modules, including the core. All variants start identically for a given seed.
- Tournament selection: best of 3, sampled with replacement.
- Subtree crossover: 90% per pair. Reject the exchange if either child exceeds 20 modules.
- Mutation: 90% per genome, with equal chances of adding a leaf, removing a subtree or replacing a node. Addition randomly chooses a free face, brick/hinge and allowed rotation. Replacement may prune incompatible branches. Impossible operations are skipped.
- Keep the best 50 parents/offspring; existing parents win ties.
- 100 generations × 50 offspring, plus initialization: 5,050 evaluations per run. Unchanged children are still evaluated. Smoke mode uses 220.

## For the report

The current mean-fitness and mean-size plots include candidates rejected that generation, not just survivors. Best-so-far is unaffected.

The size-based fitness floor is a lower bound, not a proven attainable optimum.
Use the full runs for conclusions, not the smoke test.

Run EAs sequentially or in separate processes: their random-number state is shared within a process.
