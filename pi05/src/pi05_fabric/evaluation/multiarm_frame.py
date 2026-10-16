"""Evaluation helpers shared by three- and four-arm frame tasks."""

from __future__ import annotations

import numpy as np


def dispatch_role_actions(
    role_actions: tuple[np.ndarray, ...],
    *,
    agent_count: int,
    execution_horizon: int,
) -> tuple[np.ndarray, ...]:
    if len(role_actions) != agent_count:
        raise ValueError(f"one action chunk is required from each of {agent_count} roles")
    if execution_horizon <= 0:
        raise ValueError("execution_horizon must be positive")
    dispatched = []
    for role, action in enumerate(role_actions):
        value = np.asarray(action, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] < execution_horizon:
            raise ValueError(f"role {role} action chunk is shorter than execution horizon")
        if value.shape[1] < 7:
            raise ValueError(f"role {role} action chunk has fewer than seven controls")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"role {role} produced non-finite actions")
        dispatched.append(value[:execution_horizon, :7])
    return tuple(dispatched)


__all__ = ("dispatch_role_actions",)
