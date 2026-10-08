"""The population of a generation is the one after survivor selection."""

import pandas as pd

from harness.dataset import population_after_selection


def frame(*rows):
    return pd.DataFrame(rows, columns=["id", "time_of_birth", "time_of_death", "alive"])


def ids(population):
    return sorted(population["id"])


def test_culled_children_are_not_part_of_the_population():
    """(mu + lambda): parents 1, 2; children 3 (kept) and 4 (culled) in gen 1."""
    individuals = frame(
        (1, 0, 3, False),   # parent, survives gen 1
        (2, 0, 1, False),   # parent, culled at the end of gen 1
        (3, 1, 3, False),   # child of gen 1, survives
        (4, 1, 1, False),   # child of gen 1, culled at once
    )
    assert ids(population_after_selection(individuals, 0, last=3)) == [1, 2]
    assert ids(population_after_selection(individuals, 1, last=3)) == [1, 3]


def test_last_generation_uses_the_alive_flag():
    individuals = frame(
        (1, 0, 2, True),    # still alive when the run ended
        (2, 2, 2, True),    # child of the last generation, kept
        (3, 2, 2, False),   # child of the last generation, culled
    )
    assert ids(population_after_selection(individuals, 2, last=2)) == [1, 2]
