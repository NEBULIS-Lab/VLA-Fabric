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


def checkpoint_plan(
    *,
    total_steps: int,
    permanent_steps: tuple[int, ...],
    recovery_every: int = 0,
) -> dict[int, str]:
    """Build a schedule with permanent checkpoints overriding recovery saves."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if recovery_every < 0:
        raise ValueError("recovery checkpoint interval cannot be negative")
    plan: dict[int, str] = {}
    if recovery_every:
        for step in range(recovery_every, total_steps + 1, recovery_every):
            plan[step] = "recovery"
    for step in permanent_steps:
        if step <= 0 or step > total_steps:
            raise ValueError("permanent checkpoint must be within the training horizon")
        plan[step] = "permanent"
    return dict(sorted(plan.items()))


def data_rng_for_sample(*, seed: int, step: int, sample_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, step, sample_index])
    return np.random.default_rng(sequence)
