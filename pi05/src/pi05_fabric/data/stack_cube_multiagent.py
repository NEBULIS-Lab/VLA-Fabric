"""RoboFactory ThreeRobotsStackCube data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


ROLES = ("panda-0", "panda-1", "panda-2")
WRIST_CAMERAS = (
    "wrist_camera_agent0",
    "wrist_camera_agent1",
    "wrist_camera_agent2",
)


@dataclass(frozen=True)
class QuantileStats:
    q01: np.ndarray
    q99: np.ndarray


def compute_role_quantiles(values) -> QuantileStats:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("quantile input must contain rows and features")
    if not np.isfinite(values).all():
        raise ValueError("quantile input contains non-finite values")
    return QuantileStats(
        np.quantile(values, 0.01, axis=0).astype(np.float32),
        np.quantile(values, 0.99, axis=0).astype(np.float32),
    )


def normalize_quantile(values, stats: QuantileStats):
    values = np.asarray(values, dtype=np.float32)
    return (values - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2 - 1


def unnormalize_quantile(values, stats: QuantileStats):
    values = np.asarray(values, dtype=np.float32)
    return (values + 1) / 2 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


def compute_dataset_statistics(path):
    """Compute exact role-specific state/action quantiles from training data."""

    role_values = {role: {"state": [], "action": []} for role in ROLES}
    with h5py.File(path, "r") as handle:
        for trajectory in sorted(handle.keys()):
            group = handle[trajectory]
            for role in ROLES:
                role_values[role]["state"].append(
                    np.asarray(group[f"obs/agent/{role}/qpos"][:-1], dtype=np.float32)
                )
                role_values[role]["action"].append(
                    np.asarray(group[f"actions/{role}"], dtype=np.float32)
                )
    return {
        role: {
            kind: compute_role_quantiles(np.concatenate(chunks, axis=0))
            for kind, chunks in values.items()
        }
        for role, values in role_values.items()
    }


def pad_action_horizon(actions, *, start: int, horizon: int):
    actions = np.asarray(actions)
    if actions.ndim != 2 or not 0 <= start < len(actions):
        raise ValueError("actions must be [time, dim] and start must be valid")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    chunk = actions[start : start + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.repeat(actions[-1:], horizon - len(chunk), axis=0)], axis=0
        )
    return chunk


def validate_stack_cube_h5(path, *, expected_trajectories: int):
    path = Path(path)
    with h5py.File(path, "r") as handle:
        trajectories = sorted(handle.keys())
        if len(trajectories) != expected_trajectories:
            raise ValueError(
                f"expected {expected_trajectories} trajectories, found {len(trajectories)}"
            )
        for name in trajectories:
            trajectory = handle[name]
            for role, camera in zip(ROLES, WRIST_CAMERAS, strict=True):
                action_path = f"actions/{role}"
                state_path = f"obs/agent/{role}/qpos"
                camera_path = f"obs/sensor_data/{camera}/rgb"
                for required in (action_path, state_path, camera_path):
                    if required not in trajectory:
                        raise ValueError(f"{name} is missing {required}")
                action = trajectory[action_path]
                state = trajectory[state_path]
                image = trajectory[camera_path]
                if action.ndim != 2 or action.shape[1] != 8:
                    raise ValueError(f"{name}/{action_path} must have dimension 8")
                if state.ndim != 2 or state.shape[1] != 9:
                    raise ValueError(f"{name}/{state_path} must have dimension 9")
                if len(state) != len(action) + 1 or len(image) != len(state):
                    raise ValueError(f"{name}/{role} time axes are not aligned")
            global_path = "obs/sensor_data/head_camera_global/rgb"
            if global_path not in trajectory:
                raise ValueError(f"{name} is missing {global_path}")
    return {
        "trajectory_count": len(trajectories),
        "agent_count": len(ROLES),
        "state_dimension": 9,
        "action_dimension": 8,
    }


class StackCubeMultiAgentDataset:
    """Read aligned role-local samples from the audited RoboFactory HDF5."""

    def __init__(self, path, *, action_horizon: int):
        self.path = Path(path)
        self.action_horizon = action_horizon
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        with h5py.File(self.path, "r") as handle:
            self.trajectories = tuple(sorted(handle.keys()))
            self.lengths = tuple(
                int(handle[f"{name}/actions/{ROLES[0]}"].shape[0])
                for name in self.trajectories
            )
        self.sample_count = sum(self.lengths)
        self._cumulative_lengths = np.cumsum(self.lengths)
        self._handle = None

    def sample_location(self, rng):
        flat_index = int(rng.integers(0, self.sample_count))
        trajectory_index = int(
            np.searchsorted(self._cumulative_lengths, flat_index, side="right")
        )
        previous = (
            0 if trajectory_index == 0 else int(self._cumulative_lengths[trajectory_index - 1])
        )
        return trajectory_index, flat_index - previous

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()

    def get(self, *, role_index: int, trajectory_index: int, step: int):
        if not 0 <= role_index < len(ROLES):
            raise ValueError("role_index is outside the three-agent team")
        if not 0 <= trajectory_index < len(self.trajectories):
            raise ValueError("trajectory_index is invalid")
        if not 0 <= step < self.lengths[trajectory_index]:
            raise ValueError("step is invalid")
        role = ROLES[role_index]
        camera = WRIST_CAMERAS[role_index]
        trajectory = self.trajectories[trajectory_index]
        group = self._file()[trajectory]
        actions = np.asarray(group[f"actions/{role}"], dtype=np.float32)
        return {
            "trajectory": trajectory,
            "step": step,
            "role_index": role_index,
            "role": role,
            "global_rgb": np.asarray(
                group["obs/sensor_data/head_camera_global/rgb"][step]
            ),
            "wrist_rgb": np.asarray(
                group[f"obs/sensor_data/{camera}/rgb"][step]
            ),
            "state": np.asarray(
                group[f"obs/agent/{role}/qpos"][step], dtype=np.float32
            ),
            "actions": pad_action_horizon(
                actions, start=step, horizon=self.action_horizon
            ).astype(np.float32),
        }
