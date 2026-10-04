"""Experiment code for Standard Assignment 1.

Only what is specific to this assignment lives here; the experiment machinery
(runner, dataset, statistics, shared figures) is the `harness` package in
`assignments/harness`, shared with Assignment 2.

    config.py      A1 parameters, on top of `harness.config.BaseConfig`
    fitness.py     target bodies + the assignment's official fitness
    variants.py    the tree EAs, random search, and the variant registry
    analysis.py    target-set bounds (fitness floor, size estimate), metrics
    figures.py     A1-only figures, and the order all figures are rendered in
"""

import sys
from pathlib import Path

# Make the shared `harness` package importable however this package is reached.
_ASSIGNMENTS_DIR = str(Path(__file__).resolve().parents[2])
if _ASSIGNMENTS_DIR not in sys.path:
    sys.path.insert(0, _ASSIGNMENTS_DIR)
