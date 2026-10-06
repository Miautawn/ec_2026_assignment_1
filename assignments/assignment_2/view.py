"""Watch each variant's champion in MuJoCo's interactive viewer.

    uv run assignments/assignment_2/view.py          (macOS: run with mjpython)

Uses the results of `CONFIG` in `neuroevolution/config.py`, so run the
experiment first. Shows each variant's best controller in turn, in real time
and on a loop, camera following the robot; close the window for the next one.
Videos of the same champions are written by `run_experiment.py`.
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
