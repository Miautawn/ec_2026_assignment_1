"""Put `assignments/` on sys.path so every test suite can import `harness`."""

import sys
from pathlib import Path

ASSIGNMENTS_DIR = str(Path(__file__).resolve().parent)
if ASSIGNMENTS_DIR not in sys.path:
    sys.path.insert(0, ASSIGNMENTS_DIR)
