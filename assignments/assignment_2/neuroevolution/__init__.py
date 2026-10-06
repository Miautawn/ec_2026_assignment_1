"""Neuroevolution code for Standard Assignment 2.

    config.py       every A2 parameter, on top of `harness.config.BaseConfig`
    scene.py        world + body + target, built identically for scoring and replay
    controller.py   the neural network: genome layout, inputs, forward pass
    genotype.py     weights + σ, their JSON storage, random initialisation
    provenance.py   what each individual records about its origin and evaluation
    ea.py           NeuroEA base class (the EA kit) + random search
    variants.py     the registry: which EAs the experiment runs
    metrics.py      per-generation numbers: σ, diversity, operator statistics
    replay.py       champions from the database, videos, the live viewer
    figures.py      A2-only figures (champion paths over the terrain)
    evaluator.py    genome -> one simulation -> fitness and behaviour data

Named `neuroevolution` rather than `experiment` so that it does not collide
with Assignment 1's package when both test suites run in one pytest process.
"""

import sys
from pathlib import Path

# Make the shared `harness` package importable however this package is reached.
_ASSIGNMENTS_DIR = str(Path(__file__).resolve().parents[2])
if _ASSIGNMENTS_DIR not in sys.path:
    sys.path.insert(0, _ASSIGNMENTS_DIR)
