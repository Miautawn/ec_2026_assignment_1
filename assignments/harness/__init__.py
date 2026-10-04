"""Experiment harness shared by every assignment.

Each assignment supplies its own EA variants, fitness and assignment-specific
figures; this package supplies everything around them:

    config.py     budget, seeds and output paths (subclass per assignment)
    variants.py   what a variant is, plus seeding and the shared EA settings
    runner.py     runs the (variant x seed) grid, one database per run
    dataset.py    databases -> one tidy DataFrame with a frozen schema
    analysis.py   summary table, paired significance tests, CSV report
    figures.py    plotting primitives and the figures every assignment needs

It works because every EA writes the standard ARIEL `Individual` table, so
nothing here needs to know which representation or task produced a run.

Assignments import it as `harness`, which requires `assignments/` on
`sys.path`: each assignment's `experiment/__init__.py` arranges that at
runtime, and `assignments/conftest.py` does it for the tests.
"""
