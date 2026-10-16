"""Closed-loop real-contact expert for three-arm triangular frame insertion."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from multiarm_sim.envs.triangle_frame_insertion import (
    POST_HEIGHT,
    TABLE_HEIGHT,
    THREE_ARM_BASE_XY,
)
from multiarm_sim.frame_control import FrameTargetController, world_delta_to_base


PHASES = (
    "approach",
    "descend",
    "close",
    "lift",
    "transport",
    "lower",
    "release",
    "retreat",
    "settle",
)


def eef_position_actions(
    observation: Mapping[str, np.ndarray],
    targets: np.ndarray,
    *,
    base_yaws: Sequence[float],
    grippers: Sequence[bool],
    gain: float = 8.0,
    limit: float = 0.6,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    num_agents = len(base_yaws)
    if targets.shape != (num_agents, 3):
        raise ValueError(f"targets must have shape ({num_agents}, 3), got {targets.shape}")
    if len(grippers) != num_agents:
        raise ValueError("grippers length must match base_yaws")
    actions = np.zeros((num_agents, 7), dtype=np.float64)
    for index, (target, yaw, closed) in enumerate(zip(targets, base_yaws, grippers)):
        current = np.asarray(observation[f"robot{index}_eef_pos"], dtype=np.float64)
        local = world_delta_to_base(gain * (target - current), float(yaw))
        actions[index, :3] = np.clip(local, -limit, limit)
        actions[index, 6] = 1.0 if closed else -1.0
    return actions.reshape(-1)


class ScriptedTriangleExpert:
    """State-gated expert that never moves the frame except through contacts."""

    def __init__(self, env) -> None:
        self.env = env
        self.base_yaws = tuple(
            math.atan2(-y, -x) for x, y in THREE_ARM_BASE_XY
        )
        frame_handle_offsets = env.handle_offsets()
        self.frame_controller = FrameTargetController(
            handle_offsets=frame_handle_offsets,
            base_yaws=self.base_yaws,
            position_gain=8.0,
            translation_limit=0.55,
        )
        self.phase_index = 0
        self.phase_steps = 0
        self.ready_steps = 0
        self.done = False
        self.failed = False
        self.failure_reason = ""
        self.transport_pose = env.frame_pose()
        self.transport_pose[2] = (
            TABLE_HEIGHT + POST_HEIGHT + env.frame_height / 2.0 + 0.075
        )
        self.target_pose = self.transport_pose.copy()
        self.target_pose[:2] = env.post_xy

    @property
    def phase(self) -> str:
        return PHASES[self.phase_index]

    @property
    def stage(self) -> int:
        return self.phase_index

    @property
    def next_progress(self) -> float:
        timeouts = (100, 100, 100, 150, 180, 180, 80, 100, 120)
        return min(self.phase_steps / timeouts[self.phase_index], 1.0)

    def _advance(self) -> None:
        self.phase_index = min(self.phase_index + 1, len(PHASES) - 1)
        self.phase_steps = 0
        self.ready_steps = 0

    def _advance_when(self, ready: bool, *, hold: int, timeout: int) -> None:
        self.ready_steps = self.ready_steps + 1 if ready else 0
        if self.ready_steps >= hold:
            self._advance()
        elif self.phase_steps >= timeout:
            self.failed = True
            self.done = True
            self.failure_reason = f"{self.phase} timeout"

    def action(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.done:
            idle = np.zeros((3, 7), dtype=np.float32)
            idle[:, 6] = -1.0
            return idle.reshape(-1)

        self.phase_steps += 1
        handles = np.stack(list(self.env.handle_positions().values()))
        eefs = np.stack(
            [
                np.asarray(observation[f"robot{index}_eef_pos"], dtype=np.float64)
                for index in range(3)
            ]
        )
        status = self.env.task_status()

        if self.phase == "approach":
            targets = handles + np.array([0.0, 0.0, 0.10])
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(False,) * 3
            )
            self._advance_when(
                np.max(np.linalg.norm(targets - eefs, axis=1)) < 0.030,
                hold=5,
                timeout=100,
            )
        elif self.phase == "descend":
            targets = handles + np.array([0.0, 0.0, 0.003])
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(False,) * 3
            )
            self._advance_when(
                np.max(np.linalg.norm(targets - eefs, axis=1)) < 0.035,
                hold=8,
                timeout=100,
            )
        elif self.phase == "close":
            targets = handles + np.array([0.0, 0.0, 0.003])
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(True,) * 3
            )
            if status["three_grasp_confirmed"] and self.phase_steps >= 12:
                self._advance()
            elif self.phase_steps >= 100:
                self.failed = True
                self.done = True
                self.failure_reason = f"assigned grasp timeout: {status['assigned_grasps']}"
        elif self.phase == "lift":
            action = self.frame_controller.actions(
                observation, self.transport_pose, grippers=(True,) * 3
            )
            self._advance_when(bool(status["lifted"]), hold=3, timeout=150)
        elif self.phase == "transport":
            action = self.frame_controller.actions(
                observation, self.target_pose, grippers=(True,) * 3
            )
            self._advance_when(
                np.linalg.norm(self.env.frame_pose()[:2] - self.env.post_xy) < 0.03,
                hold=6,
                timeout=180,
            )
        elif self.phase == "lower":
            lower_pose = self.target_pose.copy()
            lower_pose[2] = TABLE_HEIGHT + self.env.frame_height / 2.0 + 0.003
            action = self.frame_controller.actions(
                observation, lower_pose, grippers=(True,) * 3
            )
            self._advance_when(bool(status["placed"]), hold=6, timeout=180)
        elif self.phase == "release":
            action = self.frame_controller.actions(
                observation, self.env.frame_pose(), grippers=(False,) * 3
            )
            self._advance_when(not any(status["assigned_grasps"]), hold=12, timeout=80)
        elif self.phase == "retreat":
            frame_xy = self.env.frame_pose()[:2]
            directions = handles[:, :2] - frame_xy
            directions /= np.maximum(
                np.linalg.norm(directions, axis=1, keepdims=True), 1e-8
            )
            targets = handles.copy()
            targets[:, :2] += 0.10 * directions
            targets[:, 2] += 0.10
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(False,) * 3
            )
            self._advance_when(bool(status["retreated"]), hold=5, timeout=100)
        else:
            action = np.zeros((3, 7), dtype=np.float64)
            action[:, 6] = -1.0
            if status["success"]:
                self.done = True
            elif self.phase_steps >= 120:
                self.failed = True
                self.done = True
                self.failure_reason = "stable success timeout"
        return np.asarray(action, dtype=np.float32).reshape(-1)


__all__ = ("PHASES", "ScriptedTriangleExpert", "eef_position_actions")
