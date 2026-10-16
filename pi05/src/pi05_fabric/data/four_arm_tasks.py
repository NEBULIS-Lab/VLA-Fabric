"""Canonical PI0.5 data contracts for four-arm MuJoCo tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROLE_NAMES = tuple(f"panda-{index}" for index in range(4))


@dataclass(frozen=True)
class QuantileStats:
    q01: np.ndarray
    q99: np.ndarray


@dataclass(frozen=True)
class RoleNormalization:
    state: QuantileStats
    action: QuantileStats


@dataclass(frozen=True)
class FourArmQuantileNormalization:
    roles: tuple[RoleNormalization, ...]

    @classmethod
    def from_npz(cls, path: str | Path) -> "FourArmQuantileNormalization":
        roles = []
        with np.load(path) as arrays:
            for role in range(4):
                prefix = f"role_{role}"
                state = QuantileStats(
                    np.asarray(arrays[f"{prefix}_state_q01"], dtype=np.float32),
                    np.asarray(arrays[f"{prefix}_state_q99"], dtype=np.float32),
                )
                action = QuantileStats(
                    np.asarray(arrays[f"{prefix}_action_q01"], dtype=np.float32),
                    np.asarray(arrays[f"{prefix}_action_q99"], dtype=np.float32),
                )
                if state.q01.shape != (9,) or state.q99.shape != (9,):
                    raise ValueError(f"role {role} state quantiles must have shape (9,)")
                if action.q01.shape != (7,) or action.q99.shape != (7,):
                    raise ValueError(f"role {role} action quantiles must have shape (7,)")
                if not np.isfinite(state.q01).all() or not np.isfinite(state.q99).all():
                    raise ValueError(f"role {role} state quantiles are not finite")
                if not np.isfinite(action.q01).all() or not np.isfinite(action.q99).all():
                    raise ValueError(f"role {role} action quantiles are not finite")
                roles.append(RoleNormalization(state=state, action=action))
        return cls(tuple(roles))

    def for_role(self, role_index: int) -> RoleNormalization:
        if not 0 <= role_index < 4:
            raise ValueError("role_index is outside the four-agent team")
        return self.roles[role_index]

    def unnormalize_actions(self, role_index: int, actions: np.ndarray) -> np.ndarray:
        return unnormalize_quantile(
            np.asarray(actions)[..., :7],
            self.for_role(role_index).action,
        )


@dataclass(frozen=True)
class FourArmRoleSample:
    team_id: str
    trajectory_index: int
    step: int
    role_index: int
    global_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class FourArmTeamSample:
    team_id: str
    trajectory_index: int
    step: int
    roles: tuple[FourArmRoleSample, ...]


@dataclass(frozen=True)
class ConversionReport:
    source: Path
    output: Path
    trajectories: int
    samples: int
    source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qpos_path(trajectory: h5py.Group, role: int) -> str:
    candidates = (
        f"obs/agent/panda-{role}/qpos",
        f"obs/robot_state/panda-{role}_joint_pos",
    )
    try:
        return next(path for path in candidates if path in trajectory)
    except StopIteration as error:
        raise ValueError(f"missing qpos for role {role}") from error


def _local_cameras(trajectory: h5py.Group) -> tuple[str, ...]:
    raw = trajectory.attrs.get("local_cameras_json")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    cameras = tuple(json.loads(str(raw)))
    if len(cameras) != 4:
        raise ValueError("source trajectory must provide four wrist cameras")
    return cameras


def _copy_dataset(target: h5py.Group, name: str, source: h5py.Dataset) -> None:
    value = np.asarray(source)
    chunks = (1, *value.shape[1:]) if value.ndim >= 1 else None
    target.create_dataset(name, data=value, chunks=chunks, compression="lzf")


def convert_four_arm_release(source: str | Path, output: str | Path) -> ConversionReport:
    """Convert either canonical source schema into one task-independent layout."""

    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    sample_count = 0
    try:
        with h5py.File(source, "r") as source_handle, h5py.File(temporary, "w") as target:
            target.attrs["schema_version"] = "pi05-four-arm-v1"
            target.attrs["agent_count"] = 4
            target.attrs["source"] = str(source.resolve())
            names = sorted(key for key in source_handle if key.startswith("trajectory_"))
            if not names:
                raise ValueError("source release contains no trajectories")
            for trajectory_index, name in enumerate(names):
                if name != f"trajectory_{trajectory_index:06d}":
                    raise ValueError("source trajectory identifiers are not contiguous")
                source_trajectory = source_handle[name]
                cameras = _local_cameras(source_trajectory)
                steps = int(source_trajectory["actions/panda-0"].shape[0])
                if source_trajectory["obs/sensor_data/agentview/rgb"].shape[0] != steps:
                    raise ValueError(f"{name} agentview is not action-aligned")
                trajectory = target.create_group(name)
                trajectory.attrs["team_id"] = name
                trajectory.attrs["seed"] = int(source_trajectory.attrs["seed"])
                trajectory.attrs["instruction"] = str(source_trajectory.attrs["instruction"])
                _copy_dataset(
                    trajectory,
                    "global_rgb",
                    source_trajectory["obs/sensor_data/agentview/rgb"],
                )
                wrist_group = trajectory.create_group("wrist_rgb")
                qpos_group = trajectory.create_group("qpos")
                action_group = trajectory.create_group("actions")
                for role in range(4):
                    _copy_dataset(
                        wrist_group,
                        f"role_{role}",
                        source_trajectory[f"obs/sensor_data/{cameras[role]}/rgb"],
                    )
                    _copy_dataset(
                        qpos_group,
                        f"role_{role}",
                        source_trajectory[_qpos_path(source_trajectory, role)],
                    )
                    _copy_dataset(
                        action_group,
                        f"role_{role}",
                        source_trajectory[f"actions/panda-{role}"],
                    )
                    if qpos_group[f"role_{role}"].shape != (steps, 9):
                        raise ValueError(f"{name} role {role} qpos must be [time, 9]")
                    if action_group[f"role_{role}"].shape != (steps, 7):
                        raise ValueError(f"{name} role {role} actions must be [time, 7]")
                    if wrist_group[f"role_{role}"].shape[0] != steps:
                        raise ValueError(f"{name} role {role} wrist stream is not aligned")
                sample_count += steps
            target.attrs["trajectory_count"] = len(names)
            target.attrs["sample_count"] = sample_count
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return ConversionReport(
        source=source.resolve(),
        output=output.resolve(),
        trajectories=len(names),
        samples=sample_count,
        source_sha256=_sha256(source),
    )


def _quantiles(values: np.ndarray, *, label: str) -> QuantileStats:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite nonempty matrix")
    q01 = np.quantile(values, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(values, 0.99, axis=0).astype(np.float32)
    return QuantileStats(q01=q01, q99=q99)


def compute_four_arm_quantiles(
    converted: str | Path, output: str | Path
) -> FourArmQuantileNormalization:
    converted = Path(converted)
    arrays: dict[str, np.ndarray] = {}
    roles = []
    with h5py.File(converted, "r") as handle:
        trajectories = sorted(handle.keys())
        for role in range(4):
            states = np.concatenate(
                [np.asarray(handle[f"{name}/qpos/role_{role}"], dtype=np.float32) for name in trajectories]
            )
            actions = np.concatenate(
                [np.asarray(handle[f"{name}/actions/role_{role}"], dtype=np.float32) for name in trajectories]
            )
            state_stats = _quantiles(states, label=f"role {role} state")
            action_stats = _quantiles(actions, label=f"role {role} action")
            arrays[f"role_{role}_state_q01"] = state_stats.q01
            arrays[f"role_{role}_state_q99"] = state_stats.q99
            arrays[f"role_{role}_action_q01"] = action_stats.q01
            arrays[f"role_{role}_action_q99"] = action_stats.q99
            roles.append(RoleNormalization(state=state_stats, action=action_stats))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return FourArmQuantileNormalization(tuple(roles))


def normalize_quantile(values: np.ndarray, stats: QuantileStats) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(values: np.ndarray, stats: QuantileStats) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values + 1.0) * 0.5 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


def _pad_action_horizon(actions: np.ndarray, *, step: int, horizon: int) -> np.ndarray:
    chunk = np.asarray(actions[step : step + horizon], dtype=np.float32)
    if not len(chunk):
        raise ValueError("action step is outside the trajectory")
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)], axis=0
        )
    return chunk


class FourArmTaskDataset:
    """Read aligned four-role samples from a converted task release."""

    def __init__(self, path: str | Path, *, action_horizon: int = 50):
        self.path = Path(path)
        self.action_horizon = action_horizon
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        with h5py.File(self.path, "r") as handle:
            self.trajectories = tuple(sorted(handle.keys()))
            self.lengths = tuple(int(handle[f"{name}/actions/role_0"].shape[0]) for name in self.trajectories)
        self.sample_count = sum(self.lengths)
        self._cumulative_lengths = np.cumsum(self.lengths, dtype=np.int64)
        self._handle: h5py.File | None = None

    def sample_location(self, rng: np.random.Generator) -> tuple[int, int]:
        """Sample one aligned team timestep uniformly over all release steps."""

        flat_index = int(rng.integers(self.sample_count))
        trajectory_index = int(
            np.searchsorted(self._cumulative_lengths, flat_index, side="right")
        )
        previous = (
            0
            if trajectory_index == 0
            else int(self._cumulative_lengths[trajectory_index - 1])
        )
        return trajectory_index, flat_index - previous

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def team_sample(self, *, trajectory_index: int, step: int) -> FourArmTeamSample:
        if not 0 <= trajectory_index < len(self.trajectories):
            raise ValueError("trajectory_index is invalid")
        if not 0 <= step < self.lengths[trajectory_index]:
            raise ValueError("step is invalid")
        name = self.trajectories[trajectory_index]
        trajectory = self._file()[name]
        global_rgb = np.asarray(trajectory["global_rgb"][step])
        roles = []
        for role in range(4):
            roles.append(
                FourArmRoleSample(
                    team_id=name,
                    trajectory_index=trajectory_index,
                    step=step,
                    role_index=role,
                    global_rgb=global_rgb,
                    wrist_rgb=np.asarray(trajectory[f"wrist_rgb/role_{role}"][step]),
                    state=np.asarray(trajectory[f"qpos/role_{role}"][step], dtype=np.float32),
                    actions=_pad_action_horizon(
                        trajectory[f"actions/role_{role}"],
                        step=step,
                        horizon=self.action_horizon,
                    ),
                )
            )
        return FourArmTeamSample(
            team_id=name,
            trajectory_index=trajectory_index,
            step=step,
            roles=tuple(roles),
        )


__all__ = (
    "ConversionReport",
    "FourArmQuantileNormalization",
    "FourArmRoleSample",
    "FourArmTaskDataset",
    "FourArmTeamSample",
    "QuantileStats",
    "RoleNormalization",
    "compute_four_arm_quantiles",
    "convert_four_arm_release",
    "normalize_quantile",
    "unnormalize_quantile",
)
