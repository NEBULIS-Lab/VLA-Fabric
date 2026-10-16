"""Frozen step, sampling, and checkpoint rules for formal pi0.5 training."""

from __future__ import annotations

import numpy as np


def checkpoint_steps(*, total_steps: int, every: int) -> tuple[int, ...]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if every <= 0:
        raise ValueError("checkpoint interval must be positive")
    steps = list(range(every, total_steps + 1, every))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return tuple(steps)


def data_rng_for_sample(*, seed: int, step: int, sample_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, step, sample_index])
    return np.random.default_rng(sequence)
