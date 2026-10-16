"""Shared frame-level control utilities for the four-arm task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import robosuite.utils.transform_utils as T


SUCCESS_EVIDENCE = (
    "four_grasp_confirmed",
    "lifted",
    "transported",
    "placed",
    "released",
    "retreated",
    "stable",
)


def success_from_state(state: Mapping[str, object]) -> bool:
    """Return true only when every ordered task milestone has been observed."""

    return all(bool(state.get(name, False)) for name in SUCCESS_EVIDENCE)


def world_delta_to_base(delta_world: np.ndarray, base_yaw: float) -> np.ndarray:
    """Express a world-frame displacement in a robot's yaw-rotated base frame."""

    delta = np.asarray(delta_world, dtype=np.float64)
    if delta.shape != (3,):
        raise ValueError(f"delta_world must have shape (3,), got {delta.shape}")
    cosine = math.cos(base_yaw)
    sine = math.sin(base_yaw)
    world_from_base = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return world_from_base.T @ delta


@dataclass(frozen=True)
class FrameTargetController:
    """Map one desired rigid-frame pose to N synchronized OSC pose actions."""

    handle_offsets: np.ndarray
    base_yaws: Sequence[float]
    position_gain: float = 10.0
    translation_limit: float = 0.6
    robot_indices: Sequence[int] | None = None

    @property
    def num_agents(self) -> int:
        return int(self.handle_offsets.shape[0])

    def __post_init__(self) -> None:
        offsets = np.asarray(self.handle_offsets, dtype=np.float64)
        if offsets.ndim != 2 or offsets.shape[0] < 1 or offsets.shape[1] != 3:
            raise ValueError(f"handle_offsets must have shape (N, 3), got {offsets.shape}")
        if len(self.base_yaws) != offsets.shape[0]:
            raise ValueError("base_yaws length must match handle_offsets")
        if self.position_gain <= 0.0:
            raise ValueError("position_gain must be positive")
        if self.translation_limit <= 0.0:
            raise ValueError("translation_limit must be positive")
        indices = (
            tuple(range(offsets.shape[0]))
            if self.robot_indices is None
            else tuple(int(index) for index in self.robot_indices)
        )
        if len(indices) != offsets.shape[0] or len(set(indices)) != len(indices):
            raise ValueError("robot_indices must contain one unique index per handle")
        if any(index < 0 for index in indices):
            raise ValueError("robot_indices must be non-negative")
        object.__setattr__(self, "handle_offsets", offsets.copy())
        object.__setattr__(self, "base_yaws", tuple(float(yaw) for yaw in self.base_yaws))
        object.__setattr__(self, "robot_indices", indices)

    def actions(
        self,
        observation: Mapping[str, np.ndarray],
        target_frame_pose: np.ndarray,
        grippers: Sequence[bool],
    ) -> np.ndarray:
        """Return four concatenated ``[dpos, drot, gripper]`` actions.

        ``target_frame_pose`` is ``[x, y, z, qw, qx, qy, qz]`` in world
        coordinates. Rotation control is intentionally zero in the first version:
        all four grippers keep their established handle orientations while their
        positions realize the requested rigid-frame translation and yaw.
        """

        pose = np.asarray(target_frame_pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"target_frame_pose must have shape (7,), got {pose.shape}")
        if len(grippers) != self.num_agents:
            raise ValueError("grippers length must match handle_offsets")

        quat_xyzw = T.convert_quat(pose[3:], to="xyzw")
        rotation = T.quat2mat(quat_xyzw)
        actions = np.zeros((self.num_agents, 7), dtype=np.float64)
        for index, (robot_index, offset, base_yaw, closed) in enumerate(
            zip(self.robot_indices, self.handle_offsets, self.base_yaws, grippers)
        ):
            key = f"robot{robot_index}_eef_pos"
            if key not in observation:
                raise KeyError(f"observation is missing {key!r}")
            current = np.asarray(observation[key], dtype=np.float64)
            if current.shape != (3,):
                raise ValueError(f"{key} must have shape (3,), got {current.shape}")
            desired = pose[:3] + rotation @ offset
            local_delta = world_delta_to_base(
                self.position_gain * (desired - current), base_yaw
            )
            actions[index, :3] = np.clip(
                local_delta, -self.translation_limit, self.translation_limit
            )
            actions[index, 6] = 1.0 if closed else -1.0
        return actions.reshape(-1)
