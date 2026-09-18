# Assignment 1 — experiment harness

## Research question

> Does applying mutation to the **parents before crossover** differ from applying
> it to the **offspring after crossover**, at an identical evaluation budget?

**H:** Mutating parents before crossover lowers the *effective* mutation rate,
because subtree crossover can discard the mutated region entirely — a mutation
landing in a subtree that gets swapped out never reaches the evaluated
individual. Mutating the offspring guarantees every mutation survives into
what is scored.

Three configurations, all at the same budget:

| variant | pipeline | role |
|---|---|---|
| `mutate_child` | select → crossover → **mutate offspring** | variant A |
| `mutate_parent` | select → **mutate parent copies** → crossover | variant B |
| `random_search` | resample the whole population every generation | required baseline |

Both EA variants apply crossover at most once and mutation at most once per
produced genome, so the *amount* of variation is matched and only its ordering
differs. That is what makes this a one-aspect comparison.

## Usage

```bash
# bounds implied by the target set alone — no EA needed
python run_experiment.py targets

# fast end-to-end check of every code path (~10 s)
python run_experiment.py run --smoke

# the real thing: 3 variants x 10 seeds, ~15 min single-process
python run_experiment.py all

# re-analyse and re-render without re-running anything
python run_experiment.py analyse
```

Every invocation of `run` recomputes the whole grid from scratch. Outputs:

```
results/<variant>/seed_NN/database.db   ARIEL database (full history)
results/<variant>/seed_NN/meta.json     variant, seed, wall time, evaluations, config
tables/per_generation.csv               the tidy frame behind every figure
tables/final_per_seed.csv               one row per run: the unit of analysis
tables/summary.csv                      per-variant summary
tables/pairwise_tests.csv               Mann-Whitney U + Vargha-Delaney A12
figures/fig1,3,4,5,6.png                300 dpi, ready for the report
```

## Swapping in the real EA

`experiment/variants.py` is the **only** file that changes:

```python
from my_ea import MutateChildEA, MutateParentEA

VARIANTS = {
    "mutate_child":  MutateChildEA,
    "mutate_parent": MutateParentEA,
    "random_search": RandomSearch,   # keep: required baseline
}
```

The contract is: **an `ariel.ec.EA` subclass** whose `__init__` takes
`(seed, db_path, cfg)`. `.run()` is inherited, so there is nothing else to
implement:

```python
class MutateChildEA(EA):
    def __init__(self, seed: int, db_path: Path, cfg: ExperimentConfig) -> None:
        seed_everything(seed)
        self.cfg = cfg
        super().__init__(
            initial_population,
            operations=[...],
            num_steps=cfg.generations,
            is_maximisation=False,
            db_file_path=db_path,
            db_handling="delete",
            quiet=True,
        )
```

`variants._ea_kwargs(db_path, cfg)` bundles the settings every variant must
share, so a real class can just splat it: `super().__init__(pop,
operations=[...], **_ea_kwargs(db_path, cfg))`.

### Requirements on the real EA

1. **`is_maximisation=False`.** Fitness is lower-is-better; ARIEL defaults to
   `True`, and getting it wrong makes `get_solution("best")` return the worst
   body with no error.
2. **Write to the `db_path` you are handed.** Do not fall back to
   `config.db_file_path` — every run would overwrite the same file.
3. **Seed `random`, `numpy` *and* `torch`** before sampling the initial
   population.
4. **Hit `cfg.evaluation_budget` evaluations.** The runner reads the realised
   count back out of each database and warns loudly if the variants disagree,
   because an unequal budget invalidates the comparison.
5. **No `from __future__ import annotations` in any module defining EA
   stages.** `@EAOperation` validates signatures with
   `params[0].annotation is not Population` — an identity check against the
   real class. PEP 563 makes annotations strings, so the future import causes
   every stage to fail with `"must be annotated as Population, got
   'Population'"`.
6. **Optional but valuable:** tag offspring with
   `{"mutation_noop": bool}`. Every ARIEL tree operator silently rolls back on
   a validation failure, so a mutation can do nothing. `fig`/`mutation_noop_rate`
   measures how often — and if it is high, variant B is secretly running
   mutation-free some of the time.

## Notes on the analysis

- **The provable floor is 7.875.** Tree edit distance is a metric, so
  `d(C,A) + d(C,B) >= d(A,B)` for any candidate `C`. Summing over all target
  pairs bounds the mean distance from below, and the std term is non-negative.
  No body can score better than this. It is drawn on every fitness figure.
- **`best_so_far`, not `best_fitness`, is the headline metric.** Random search
  has no population continuity, so cumulative best is the only fair comparison.
- **N is the number of seeds, not the number of individuals.** Every test uses
  one observation per independent run.
- **Two different "mean + std" appear in this assignment.** Inside the fitness
  it is across the 5 *targets* for one individual — part of the objective.
  On the figures it is across *runs* — descriptive statistics. They are never
  the same thing and the error bands are always the latter.
- The course's `plot_fit_per_gen.py` filters generations with
  `time_of_death > gen`; it should be `>=`, otherwise the final generation's
  bucket is empty and silently dropped. `dataset.py` uses `>=`.
