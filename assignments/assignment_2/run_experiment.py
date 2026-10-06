"""Run the configured experiment, then write its tables, figures and videos.

    uv run assignments/assignment_2/run_experiment.py

What runs is `CONFIG` in `neuroevolution/config.py`: the variants, the seeds,
the budget, the world, and how many runs execute in parallel (`workers`).
Edit it there and re-run. Results land in `outputs/<name>/`, including a
video of each variant's champion in `outputs/<name>/videos/`.
"""

import sys
from pathlib import Path

# The local package and the shared harness.
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from harness import dataset, figures, runner  # noqa: E402
from harness.analysis import report  # noqa: E402
from neuroevolution import figures as a2_figures  # noqa: E402
from neuroevolution import replay  # noqa: E402
from neuroevolution.config import CONFIG  # noqa: E402
from neuroevolution.metrics import METRICS, POPULATION_METRICS  # noqa: E402
from neuroevolution.variants import VARIANTS  # noqa: E402


def main() -> None:
    runner.run_all(CONFIG, VARIANTS)

    tidy = dataset.load(CONFIG, METRICS, POPULATION_METRICS)
    final, _, _ = report(tidy, CONFIG)
    figures.convergence(tidy, VARIANTS, CONFIG.figures_dir / "convergence.png")
    figures.final_distribution(final, VARIANTS, CONFIG.figures_dir / "final_distribution.png")

    champions = replay.champions(CONFIG)
    a2_figures.trajectories(CONFIG, champions, VARIANTS, CONFIG.figures_dir / "trajectories.png")
    for variant, champion in champions.items():
        replay.video(CONFIG, champion, replay.videos_dir(CONFIG) / f"{variant}.mp4")


if __name__ == "__main__":  # required: worker processes re-import this file
    main()
