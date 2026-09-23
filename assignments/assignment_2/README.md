# Assignment 2: Neuroevolution

Group 64.

## Requirements

- Due **13 October 2026, 09:00** (Amsterdam time).
- 10 points; file upload. Same grading rubric as Assignment 1.
- GECCO19 report template; maximum **6 pages**, excluding cover and bibliography.
- Submit `64.zip` containing `64/64.pdf` and the code.
- Evolve neural-network weights using an EA built on `ariel.ec`.
- Fix one John Set body and one supported environment throughout the assignment. No CPG controller or `SimpleTiltedWorld`.
- Default fitness: final ground-plane distance to the fixed target, lower is better.
- Investigate one EA aspect. Compare against a baseline at equal evaluation budget, with at least five independent repeats per configuration and mean/SD convergence plots.
- Do not edit `src/ariel` or use a black-box optimiser in place of the team's EA.

See [the brief](reference/Assignment2.pdf) for the full requirements. Course rules on contributions and disclosure still apply.

## Layout

```text
A2_template_2026.py   official upstream demo (random weights, no evolution)
reference/   official PDF brief
tests/       framework compatibility and demo smoke tests
outputs/     generated files only; ignored except the README
```

## Template compatibility

The [official demo](A2_template_2026.py) and its required framework update are now included unchanged from upstream. It is runnable, but is not an EA implementation.

Source: [upstream commit a435b192](https://github.com/AndrzejSzczepura/EvolutionaryComputing2026/blob/a435b192066d258096eba3f43f91c43b74cf11fe/assignments/assignment_2/A2_template_2026.py), inspected 23 September 2026.

- The official update adds `ariel.ec.set_seed` and removes random initial control values from the headless runner.
- It defaults to `gecko()`, whereas the brief requires a John Set body. Follow the brief or ask the TA before using the demo body.
- Its default mode launches a viewer. Final search evaluations need headless execution.
- It evaluates random weights only; it does not implement an EA.

No team-written changes were made to `src/ariel`. Assignment 1 code and saved results are unchanged.

From the repository root:

```sh
uv run pytest assignments/assignment_1/tests assignments/assignment_2/tests
# Interactive demo (opens a viewer):
uv run assignments/assignment_2/A2_template_2026.py
```

The tests run the demo headlessly with a short episode and check seed replay. They do not establish controller quality or replace final experiments.
