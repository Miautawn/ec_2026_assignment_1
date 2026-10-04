"""
A variant pairs an EA factory with the label and colour every figure uses for
it, so the registry is one dict rather than three parallel ones. The factory
must accept `(seed, db_path, cfg)` and return an `ariel.ec.EA`; `.run()` is
inherited from `EA`, so that is the whole contract.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ariel.ec import EA, set_seed

from .config import BaseConfig


@dataclass(frozen=True)
class Variant:
    """One registered EA configuration and how figures should show it."""

    factory: Callable[..., EA]
    label: str
    colour: str

    def build(self, seed: int, db_path: Path, cfg: BaseConfig) -> EA:
        return self.factory(seed=seed, db_path=db_path, cfg=cfg)


def seed_everything(seed: int) -> None:
    """Seed every RNG an EA built on ARIEL may draw from.

    `random` (tree operators), numpy, torch (NDE networks) and ARIEL's own
    package-level RNG behind `ariel.ec`'s generators, mutators and crossover.
    Missing the last one means every seed runs the same variation operators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def ea_settings(db_path: Path, cfg: BaseConfig, *, is_maximisation: bool) -> dict:
    """`EA.__init__` keyword arguments every variant must share.

    `is_maximisation` is deliberately required: ARIEL defaults to `True`, and
    getting it wrong on a lower-is-better fitness silently makes
    `get_solution("best")` return the worst individual.
    """
    return {
        "num_steps": cfg.generations,
        "first_generation_id": 0,
        "is_maximisation": is_maximisation,
        "db_file_path": db_path,
        "db_handling": "delete",  # the runner guarantees a clean directory
        "quiet": True,
    }
