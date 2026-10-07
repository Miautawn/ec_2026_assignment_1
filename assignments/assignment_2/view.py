"""Shows each EA's best robot in the 3D viewer; close the window for the next.

    uv run assignments/assignment_2/view.py      (on macOS: mjpython)

Uses the results of `CONFIG`, so run the experiment first.
"""

import sys
from pathlib import Path

# The local package and the shared harness.
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from neuroevolution import replay  # noqa: E402
from neuroevolution.config import CONFIG  # noqa: E402


def main() -> None:
    champions = replay.champions(CONFIG)
    if not champions:
        print(f"No runs under {CONFIG.results_dir}: run run_experiment.py first.")
        return
    for variant, champion in champions.items():
        print(f"{variant}: seed {champion.seed}, fitness {champion.fitness:.3f} "
              "- close the window to continue")
        replay.view(CONFIG, champion)


if __name__ == "__main__":
    main()
