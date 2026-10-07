"""Assignment 2: evolving neural-network controllers for a John Set robot.

Start with EXPERIMENT.md. In short: config.py says what to run, ea.py is where
EAs are written, variants.py lists them; the rest is plumbing.

(Named `neuroevolution`, not `experiment`, so it cannot clash with
Assignment 1's package when both test suites run together.)
"""

import sys
from pathlib import Path

# Make the shared `harness` package importable however this package is reached.
_ASSIGNMENTS_DIR = str(Path(__file__).resolve().parents[2])
if _ASSIGNMENTS_DIR not in sys.path:
    sys.path.insert(0, _ASSIGNMENTS_DIR)
