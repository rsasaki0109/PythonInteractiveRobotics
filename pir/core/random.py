"""Random number helpers."""

from __future__ import annotations

import random

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy generator from an optional seed."""

    return np.random.default_rng(seed)


def seed_python_and_numpy(seed: int | None = None) -> np.random.Generator:
    """Seed Python's random module and return a NumPy generator."""

    if seed is not None:
        random.seed(seed)
    return make_rng(seed)
