"""Smoke outputs must never overwrite final results.

The paired-test tests moved with the code, to assignments/harness/tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.config import ExperimentConfig, SMOKE


def test_smoke_does_not_overwrite_final_outputs():
    full = ExperimentConfig()
    assert SMOKE.results_dir.parent != full.results_dir.parent
    assert SMOKE.figures_dir != full.figures_dir
