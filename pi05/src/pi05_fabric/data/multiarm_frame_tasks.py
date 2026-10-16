"""Canonical PI0.5 data contracts for three- and four-arm frame tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class QuantileStats:
    q01: np.ndarray
    q99: np.ndarray


@dataclass(frozen=True)
class RoleNormalization:
    state: QuantileStats
    action: QuantileStats


@dataclass(frozen=True)
class MultiArmQuantileNormalization:
    roles: tuple[RoleNormalization, ...]

    @property
    def agent_count(self) -> int:
        return len(self.roles)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_agent_count: int,
    ) -> "MultiArmQuantileNormalization":
        roles = []
        with np.load(path) as arrays:
            for role in range(expected_agent_count):
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
                if not all(
                    np.isfinite(value).all()
                    for value in (state.q01, state.q99, action.q01, action.q99)
                ):
                    raise ValueError(f"role {role} quantiles are not finite")
                roles.append(RoleNormalization(state=state, action=action))
            unexpected = [
                key
                for key in arrays.files
                if key.startswith("role_")
                and int(key.split("_", 2)[1]) >= expected_agent_count
            ]
            if unexpected:
                raise ValueError("normalization contains roles outside the requested team")
        return cls(tuple(roles))

    def for_role(self, role_index: int) -> RoleNormalization:
        if not 0 <= role_index < self.agent_count:
            raise ValueError("role_index is outside the multi-agent team")
        return self.roles[role_index]

    def unnormalize_actions(self, role_index: int, actions: np.ndarray) -> np.ndarray:
        return unnormalize_quantile(
            np.asarray(actions)[..., :7],
            self.for_role(role_index).action,
        )


@dataclass(frozen=True)
class MultiArmRoleSample:
    team_id: str
    trajectory_index: int
    step: int
    role_index: int
    global_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class MultiArmTeamSample:
    team_id: str
    trajectory_index: int
    step: int
    roles: tuple[MultiArmRoleSample, ...]


@dataclass(frozen=True)
class ConversionReport:
    source: Path
    output: Path
    agent_count: int
    trajectories: int
    samples: int
    source_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _qpos_path(trajectory: h5py.Group, role: int) -> str:
    candidates = (
        f"obs/agent/panda-{role}/qpos",
        f"obs/robot_state/panda-{role}_joint_pos",
    )
    try:
        return next(path for path in candidates if path in trajectory)
    except StopIteration as error:
        raise ValueError(f"missing qpos for role {role}") from error


def _local_cameras(trajectory: h5py.Group, *, agent_count: int) -> tuple[str, ...]:
    raw = _decoded(trajectory.attrs.get("local_cameras_json", "[]"))
    cameras = tuple(json.loads(str(raw)))
    if len(cameras) != agent_count:
        raise ValueError(
            f"source trajectory must provide {agent_count} wrist cameras"
        )
    return cameras


def _copy_dataset(target: h5py.Group, name: str, source: h5py.Dataset) -> None:
    value = np.asarray(source)
    chunks = (1, *value.shape[1:]) if value.ndim >= 1 else None
    target.create_dataset(name, data=value, chunks=chunks, compression="lzf")


def _validate_source_root(source: h5py.File, *, agent_count: int) -> None:
    if int(source.attrs.get("num_agents", -1)) != agent_count:
        raise ValueError("source num_agents does not match the requested team")
    if not bool(source.attrs.get("training_ready", False)):
        raise ValueError("source release is not marked training_ready")
    if bool(source.attrs.get("preview_only", True)):
        raise ValueError("preview-only data cannot be converted for formal training")


def convert_multiarm_release(
    source: str | Path,
    output: str | Path,
    *,
    agent_count: int,
) -> ConversionReport:
    """Convert a formal 3/4-arm release into one task-independent layout."""

    if agent_count not in (3, 4):
        raise ValueError("only three- and four-agent releases are supported")
    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    sample_count = 0
    names: list[str] = []
    try:
        with h5py.File(source, "r") as source_handle, h5py.File(temporary, "w") as target:
            _validate_source_root(source_handle, agent_count=agent_count)
            target.attrs["schema_version"] = "pi05-multiarm-v1"
            target.attrs["agent_count"] = agent_count
            target.attrs["source"] = str(source.resolve())
            names = sorted(key for key in source_handle if key.startswith("trajectory_"))
            if not names:
                raise ValueError("source release contains no trajectories")
            for trajectory_index, name in enumerate(names):
                if name != f"trajectory_{trajectory_index:06d}":
                    raise ValueError("source trajectory identifiers are not contiguous")
                source_trajectory = source_handle[name]
                if not bool(source_trajectory.attrs.get("success", False)):
                    raise ValueError(f"{name} is not a successful demonstration")
                if not bool(source_trajectory.attrs.get("complete", False)):
                    raise ValueError(f"{name} is not complete")
                cameras = _local_cameras(source_trajectory, agent_count=agent_count)
                steps = int(source_trajectory["actions/panda-0"].shape[0])
                if source_trajectory["obs/sensor_data/agentview/rgb"].shape[0] != steps:
                    raise ValueError(f"{name} agentview is not action-aligned")
                trajectory = target.create_group(name)
                trajectory.attrs["team_id"] = name
                trajectory.attrs["seed"] = int(source_trajectory.attrs["seed"])
                trajectory.attrs["instruction"] = str(
                    _decoded(source_trajectory.attrs["instruction"])
                )
                _copy_dataset(
                    trajectory,
                    "global_rgb",
                    source_trajectory["obs/sensor_data/agentview/rgb"],
                )
                wrist_group = trajectory.create_group("wrist_rgb")
                qpos_group = trajectory.create_group("qpos")
                action_group = trajectory.create_group("actions")
                for role in range(agent_count):
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
        temporary.unlink(missing_ok=True)
        raise
    return ConversionReport(
        source=source.resolve(),
        output=output.resolve(),
        agent_count=agent_count,
        trajectories=len(names),
        samples=sample_count,
        source_sha256=sha256_file(source),
    )


def _quantiles(values: np.ndarray, *, label: str) -> QuantileStats:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite nonempty matrix")
    return QuantileStats(
        q01=np.quantile(values, 0.01, axis=0).astype(np.float32),
        q99=np.quantile(values, 0.99, axis=0).astype(np.float32),
    )


def compute_multiarm_quantiles(
    converted: str | Path,
    output: str | Path,
    *,
    expected_agent_count: int,
) -> MultiArmQuantileNormalization:
    arrays: dict[str, np.ndarray] = {}
    roles = []
    with h5py.File(converted, "r") as handle:
        if int(handle.attrs.get("agent_count", -1)) != expected_agent_count:
            raise ValueError("converted dataset agent count mismatch")
        trajectories = sorted(handle.keys())
        for role in range(expected_agent_count):
            states = np.concatenate(
                [
                    np.asarray(handle[f"{name}/qpos/role_{role}"], dtype=np.float32)
                    for name in trajectories
                ]
            )
            actions = np.concatenate(
                [
                    np.asarray(handle[f"{name}/actions/role_{role}"], dtype=np.float32)
                    for name in trajectories
                ]
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
    return MultiArmQuantileNormalization(tuple(roles))


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


class MultiArmTaskDataset:
    """Read aligned role samples from a converted 3/4-arm release."""

    def __init__(
        self,
        path: str | Path,
        *,
        agent_count: int,
        action_horizon: int = 50,
    ):
        self.path = Path(path)
        self.agent_count = agent_count
        self.action_horizon = action_horizon
        if agent_count not in (3, 4):
            raise ValueError("agent_count must be three or four")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        with h5py.File(self.path, "r") as handle:
            if int(handle.attrs.get("agent_count", -1)) != agent_count:
                raise ValueError("dataset agent count mismatch")
            self.trajectories = tuple(sorted(handle.keys()))
            self.lengths = tuple(
                int(handle[f"{name}/actions/role_0"].shape[0])
                for name in self.trajectories
            )
        self.sample_count = sum(self.lengths)
        self._cumulative_lengths = np.cumsum(self.lengths, dtype=np.int64)
        self._handle: h5py.File | None = None

    def sample_location(self, rng: np.random.Generator) -> tuple[int, int]:
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

    def team_sample(self, *, trajectory_index: int, step: int) -> MultiArmTeamSample:
        if not 0 <= trajectory_index < len(self.trajectories):
            raise ValueError("trajectory_index is invalid")
        if not 0 <= step < self.lengths[trajectory_index]:
            raise ValueError("step is invalid")
        name = self.trajectories[trajectory_index]
        trajectory = self._file()[name]
        global_rgb = np.asarray(trajectory["global_rgb"][step])
        roles = []
        for role in range(self.agent_count):
            roles.append(
                MultiArmRoleSample(
                    team_id=name,
                    trajectory_index=trajectory_index,
                    step=step,
                    role_index=role,
                    global_rgb=global_rgb,
                    wrist_rgb=np.asarray(trajectory[f"wrist_rgb/role_{role}"][step]),
                    state=np.asarray(
                        trajectory[f"qpos/role_{role}"][step], dtype=np.float32
                    ),
                    actions=_pad_action_horizon(
                        trajectory[f"actions/role_{role}"],
                        step=step,
                        horizon=self.action_horizon,
                    ),
                )
            )
        return MultiArmTeamSample(
            team_id=name,
            trajectory_index=trajectory_index,
            step=step,
            roles=tuple(roles),
        )


__all__ = (
    "ConversionReport",
    "MultiArmQuantileNormalization",
    "MultiArmRoleSample",
    "MultiArmTaskDataset",
    "MultiArmTeamSample",
    "QuantileStats",
    "RoleNormalization",
    "compute_multiarm_quantiles",
    "convert_multiarm_release",
    "normalize_quantile",
    "sha256_file",
    "unnormalize_quantile",
)
