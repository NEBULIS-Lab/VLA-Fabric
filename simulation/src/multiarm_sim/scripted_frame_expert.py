"""Deterministic real-contact scripted expert for four-arm frame insertion."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from multiarm_sim.envs.frame_insertion import ROBOT_BASE_XY, TABLE_HEIGHT
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
    """Drive four EEF positions while leaving their safe top-down orientation fixed."""

    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (4, 3):
        raise ValueError(f"targets must have shape (4, 3), got {targets.shape}")
    if len(base_yaws) != 4 or len(grippers) != 4:
        raise ValueError("exactly four base yaws and gripper states are required")
    actions = np.zeros((4, 7), dtype=np.float64)
    for index, (target, yaw, closed) in enumerate(zip(targets, base_yaws, grippers)):
        current = np.asarray(observation[f"robot{index}_eef_pos"], dtype=np.float64)
        local = world_delta_to_base(gain * (target - current), float(yaw))
        actions[index, :3] = np.clip(local, -limit, limit)
        actions[index, 6] = 1.0 if closed else -1.0
    return actions.reshape(-1)


class ScriptedFrameExpert:
    """Closed-loop staged policy; it never welds or teleports the frame."""

    def __init__(self, env) -> None:
        self.env = env
        self.base_yaws = tuple(math.atan2(-y, -x) for x, y in ROBOT_BASE_XY)
        self.frame_controller = FrameTargetController(
            handle_offsets=env.handle_offsets(),
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
        self.transport_pose[2] = TABLE_HEIGHT + env.post_height + env.frame_height / 2 + 0.075
        self.target_pose = self.transport_pose.copy()
        self.target_pose[:2] = 0.0

    @property
    def phase(self) -> str:
        return PHASES[self.phase_index]

    @property
    def stage(self) -> int:
        return self.phase_index

    def _advance(self) -> None:
        self.phase_index = min(self.phase_index + 1, len(PHASES) - 1)
        self.phase_steps = 0
        self.ready_steps = 0

    def _advance_when(self, ready: bool, *, hold: int, timeout: int) -> None:
        self.ready_steps = self.ready_steps + 1 if ready else 0
        if self.ready_steps >= hold or self.phase_steps >= timeout:
            self._advance()

    def action(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.done:
            return np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], 4)

        self.phase_steps += 1
        handles = np.stack(list(self.env.handle_positions().values()))
        eefs = np.stack(
            [np.asarray(observation[f"robot{i}_eef_pos"], dtype=np.float64) for i in range(4)]
        )
        status = self.env.task_status()

        if self.phase == "approach":
            targets = handles + np.array([0.0, 0.0, 0.10])
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(False,) * 4
            )
            self._advance_when(
                np.max(np.linalg.norm(targets - eefs, axis=1)) < 0.012,
                hold=5,
                timeout=100,
            )
        elif self.phase == "descend":
            targets = handles + np.array([0.0, 0.0, 0.003])
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(False,) * 4
            )
            self._advance_when(
                np.max(np.linalg.norm(targets - eefs, axis=1)) < 0.008,
                hold=8,
                timeout=100,
            )
        elif self.phase == "close":
            targets = handles + np.array([0.0, 0.0, 0.003])
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(True,) * 4
            )
            if all(status["assigned_grasps"]) and self.phase_steps >= 12:
                self._advance()
            elif self.phase_steps >= 100:
                self.failed = True
                self.done = True
                self.failure_reason = f"assigned grasp timeout: {status['assigned_grasps']}"
        elif self.phase == "lift":
            action = self.frame_controller.actions(
                observation, self.transport_pose, grippers=(True,) * 4
            )
            self._advance_when(bool(status["lifted"]), hold=3, timeout=140)
        elif self.phase == "transport":
            action = self.frame_controller.actions(
                observation, self.target_pose, grippers=(True,) * 4
            )
            self._advance_when(
                np.linalg.norm(self.env.frame_pose()[:2]) < 0.025,
                hold=6,
                timeout=180,
            )
        elif self.phase == "lower":
            lower_pose = self.target_pose.copy()
            lower_pose[2] = TABLE_HEIGHT + self.env.frame_height / 2 + 0.002
            action = self.frame_controller.actions(
                observation, lower_pose, grippers=(True,) * 4
            )
            self._advance_when(bool(status["placed"]), hold=6, timeout=180)
        elif self.phase == "release":
            action = self.frame_controller.actions(
                observation, self.env.frame_pose(), grippers=(False,) * 4
            )
            self._advance_when(not any(status["assigned_grasps"]), hold=12, timeout=80)
        elif self.phase == "retreat":
            frame_xy = self.env.frame_pose()[:2]
            directions = handles[:, :2] - frame_xy
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
            targets = handles.copy()
            targets[:, :2] += 0.10 * directions
            targets[:, 2] += 0.10
            action = eef_position_actions(
                observation, targets, base_yaws=self.base_yaws, grippers=(False,) * 4
            )
            self._advance_when(bool(status["retreated"]), hold=5, timeout=100)
        else:
            action = np.tile(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], 4
            ).astype(np.float64)
            if status["success"]:
                self.done = True
            elif self.phase_steps >= 120:
                self.failed = True
                self.done = True
                self.failure_reason = "stable success timeout"
        return np.asarray(action, dtype=np.float32)
