"""Strict three-policy evaluation contracts for ThreeRobotsStackCube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class StackCubeProgress:
    cube_b_on_a: bool
    cube_c_on_b: bool
    cube_b_in_goal: bool
    cube_c_in_goal: bool
    role_grasping_own_cube: tuple[bool, bool, bool]


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


def strict_success(progress: StackCubeProgress) -> bool:
    if len(progress.role_grasping_own_cube) != 3:
        raise ValueError("strict success requires one release state per role")
    return bool(
        progress.cube_b_on_a
        and progress.cube_c_on_b
        and progress.cube_b_in_goal
        and progress.cube_c_in_goal
        and not any(progress.role_grasping_own_cube)
    )


def dispatch_role_actions(
    role_actions: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    execution_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(role_actions) != 3:
        raise ValueError("one action chunk is required from each of three roles")
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


def evaluation_seeds(split: Literal["validation", "fresh"]) -> tuple[int, ...]:
    if split == "validation":
        return tuple(range(40000, 40200))
    if split == "fresh":
        return tuple(range(40200, 40400))
    raise ValueError(f"unsupported evaluation split: {split}")
