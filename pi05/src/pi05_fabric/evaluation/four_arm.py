"""Strict four-policy evaluation contracts for four-arm MuJoCo tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FourArmProgress:
    task: Literal["frame_insertion", "arch_assembly"]
    task_state: dict[str, object]


@dataclass(frozen=True)
class RolePolicyRequest:
    seed: int
    planning_round: int
    role_index: int
    global_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    instruction: str


@dataclass(frozen=True)
class RoleActionResponse:
    seed: int
    planning_round: int
    role_index: int
    actions: np.ndarray


def strict_success(progress: FourArmProgress) -> bool:
    if progress.task not in {"frame_insertion", "arch_assembly"}:
        raise ValueError("unsupported four-arm task")
    return bool(progress.task_state.get("success", False))


def dispatch_role_actions(
    role_actions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    execution_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(role_actions) != 4:
        raise ValueError("one action chunk is required from each of four roles")
    if execution_horizon <= 0:
        raise ValueError("execution_horizon must be positive")
    dispatched = []
    for role, action in enumerate(role_actions):
        value = np.asarray(action, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] < execution_horizon:
            raise ValueError(f"role {role} action chunk is shorter than execution horizon")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"role {role} produced non-finite actions")
        dispatched.append(value[:execution_horizon])
    return tuple(dispatched)


def evaluation_seeds(
    task: Literal["frame_insertion", "arch_assembly"],
    split: Literal["validation", "fresh"],
) -> tuple[int, ...]:
    starts = {
        ("frame_insertion", "validation"): 5100,
        ("frame_insertion", "fresh"): 5300,
        ("arch_assembly", "validation"): 6200,
        ("arch_assembly", "fresh"): 6400,
    }
    try:
        start = starts[(task, split)]
    except KeyError as error:
        raise ValueError(f"unsupported evaluation split: {task} {split}") from error
    return tuple(range(start, start + 200))
