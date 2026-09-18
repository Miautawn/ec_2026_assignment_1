"""The assignment's official fitness, and the target set it is measured against.

This module is the *only* place that imports `tree_edit_distance`, so if the
cost constants or the aggregation ever change there is exactly one place to
look. The metric itself is never modified: the brief says not to touch the
cost weights unless they are the subject of the research question.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import networkx as nx

from .config import ASSIGNMENT_DIR

# `tree_edit_distance.py` is a flat module beside this package, not an
# installed one, so put its directory on the path once, here.
if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))

from tree_edit_distance import (  # noqa: E402
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (  # noqa: E402
    load_graph_from_json,
)

TARGET_DIR: Path = ASSIGNMENT_DIR / "target_bodies"

#: Lower is better, and the EA must be told so (`is_maximisation=False`).
IS_MAXIMISATION: bool = False


@functools.lru_cache(maxsize=1)
def load_targets(target_dir: Path = TARGET_DIR) -> tuple[nx.DiGraph, ...]:
    """Load every target body, sorted by filename.

    Cached, because the targets are immutable and every evaluation needs them.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return tuple(load_graph_from_json(path) for path in paths)


def evaluate_body(body: nx.DiGraph) -> float:
    """Score one decoded body against the whole target set. LOWER IS BETTER.

    Mean tree edit distance across all targets, plus one population standard
    deviation of those per-target distances.
    """
    return mean_plus_std_tree_edit_distance(body, list(load_targets()))


def per_target_distances(body: nx.DiGraph) -> list[float]:
    """Per-target breakdown for one body.

    The aggregated fitness hides *which* target a compromise body sacrifices;
    this is what reveals it, and it belongs in the report's discussion.
    """
    return list(distances_to_targets(body, list(load_targets())))


__all__ = [
    "IS_MAXIMISATION",
    "TARGET_DIR",
    "evaluate_body",
    "load_targets",
    "per_target_distances",
    "tree_edit_distance",
]
