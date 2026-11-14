"""Randomness controls shared across experiment commands."""

from __future__ import annotations

import os
import random


DEFAULT_SEED = 20260613


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Set stdlib-level randomness for deterministic phase-1 commands."""

    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed
