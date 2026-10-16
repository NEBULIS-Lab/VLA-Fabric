"""Model-neutral synchronous HDF5 storage for an arbitrary number of arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
import uuid

import cv2
import h5py
import numpy as np


SCHEMA_VERSION = "multiarm-sim-hdf5-v3"


@dataclass(frozen=True)
class MultiArmDatasetSpec:
    environment: str
    num_agents: int
    global_camera: str
    local_cameras: tuple[str, ...]
    role_instructions: tuple[str, ...]
    control_frequency_hz: int = 20

    def __post_init__(self) -> None:
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if len(self.local_cameras) != self.num_agents:
            raise ValueError("one local camera is required for every agent")
        if len(self.role_instructions) != self.num_agents:
            raise ValueError("one role instruction is required for every agent")
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")


def _top_left_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(np.asarray(image, dtype=np.uint8)))


def _proprio(observation: dict[str, np.ndarray], index: int) -> np.ndarray:
    return np.concatenate(
        [
            observation[f"robot{index}_joint_pos"],
            observation[f"robot{index}_gripper_qpos"],
        ]
    ).astype(np.float32)


@dataclass
class MultiArmEpisodeBuffer:
    """Aligned pre-action observations and synchronous multi-arm commands."""

    spec: MultiArmDatasetSpec
    seed: int
    source: str
    instruction: str
    episode_metadata: dict[str, Any] = field(default_factory=dict)
    recording_policy: str = "continuous"
    motion_tail_seconds: float = 0.0
    gripper_tail_seconds: float = 0.0
    actions: list[np.ndarray] = field(default_factory=list)
    proprio: list[list[np.ndarray]] = field(init=False)
    global_images: list[np.ndarray] = field(default_factory=list)
    local_images: list[list[np.ndarray]] = field(init=False)
    sim_states: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    stages: list[int] = field(default_factory=list)
    active_arms: list[int] = field(default_factory=list)
    control_modes: list[str] = field(default_factory=list)
    grasp_flags: list[np.ndarray] = field(default_factory=list)
    wall_timestamps: list[float] = field(default_factory=list)
    capture_reasons: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.proprio = [[] for _ in range(self.spec.num_agents)]
        self.local_images = [[] for _ in range(self.spec.num_agents)]

    def append(
        self,
        *,
        observation: dict[str, np.ndarray],
        sim_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        stage: int,
        active_arm: int,
        control_mode: str,
        grasp_flags: Sequence[bool],
        wall_timestamp: float | None = None,
        capture_reason: int = 0,
    ) -> None:
        action = np.asarray(action, dtype=np.float32)
        expected = self.spec.num_agents * 7
        if action.shape != (expected,):
            raise ValueError(f"action must have shape ({expected},), got {action.shape}")
        if len(grasp_flags) != self.spec.num_agents:
            raise ValueError("grasp_flags length must match num_agents")
        if wall_timestamp is None:
            wall_timestamp = len(self) / self.spec.control_frequency_hz

        self.actions.append(action.copy())
        for index, camera in enumerate(self.spec.local_cameras):
            self.proprio[index].append(_proprio(observation, index))
            self.local_images[index].append(_top_left_rgb(observation[f"{camera}_image"]))
        self.global_images.append(
            _top_left_rgb(observation[f"{self.spec.global_camera}_image"])
        )
        self.sim_states.append(np.asarray(sim_state, dtype=np.float64).copy())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.successes.append(bool(success))
        self.stages.append(int(stage))
        self.active_arms.append(int(active_arm))
        self.control_modes.append(str(control_mode))
        self.grasp_flags.append(np.asarray(grasp_flags, dtype=np.bool_))
        self.wall_timestamps.append(float(wall_timestamp))
        self.capture_reasons.append(int(capture_reason))

    def __len__(self) -> int:
        return len(self.actions)


def _trajectory_names(handle: h5py.File) -> list[str]:
    return sorted(
        name
        for name in handle
        if name.startswith("trajectory_") and name.removeprefix("trajectory_").isdigit()
    )


def _next_trajectory_name(handle: h5py.File) -> str:
    indices = [int(name.removeprefix("trajectory_")) for name in _trajectory_names(handle)]
    return f"trajectory_{max(indices, default=-1) + 1:06d}"


def _dataset(group: h5py.Group, name: str, values: np.ndarray, *, images=False) -> None:
    kwargs = {}
    if len(values):
        kwargs = {
            "compression": "gzip",
            "compression_opts": 4 if images else 1,
            "shuffle": True,
        }
    group.create_dataset(name, data=values, **kwargs)


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=lambda x: np.asarray(x).tolist())


def append_multiarm_episode(
    path: str | Path,
    episode: MultiArmEpisodeBuffer,
    *,
    final_sim_state: np.ndarray,
    success: bool,
) -> str:
    """Append an episode atomically so readers never see partial trajectories."""

    if not len(episode):
        raise ValueError("Cannot save an empty episode")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = episode.spec
    with h5py.File(path, "a") as handle:
        existing_agents = handle.attrs.get("num_agents")
        if existing_agents is not None and int(existing_agents) != spec.num_agents:
            raise ValueError(f"{path} already contains a {existing_agents}-agent dataset")
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["simulator"] = "MuJoCo"
        handle.attrs["environment"] = spec.environment
        handle.attrs["num_agents"] = spec.num_agents

        name = _next_trajectory_name(handle)
        temporary_name = f"_writing_{uuid.uuid4().hex}"
        trajectory = handle.create_group(temporary_name)
        trajectory.attrs["complete"] = False
        trajectory.attrs["success"] = bool(success)
        trajectory.attrs["seed"] = int(episode.seed)
        trajectory.attrs["source"] = episode.source
        trajectory.attrs["instruction"] = episode.instruction
        trajectory.attrs["control_frequency_hz"] = spec.control_frequency_hz
        trajectory.attrs["action_semantics"] = "delta_eef_xyz_axis_angle_and_gripper"
        trajectory.attrs["action_gripper"] = "+1 close, -1 open"
        trajectory.attrs["image_origin"] = "top_left"
        trajectory.attrs["global_camera"] = spec.global_camera
        trajectory.attrs["local_cameras_json"] = json.dumps(spec.local_cameras)
        trajectory.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        trajectory.attrs["recording_policy"] = episode.recording_policy
        trajectory.attrs["motion_tail_seconds"] = episode.motion_tail_seconds
        trajectory.attrs["gripper_tail_seconds"] = episode.gripper_tail_seconds
        for key, value in episode.episode_metadata.items():
            trajectory.attrs[key] = _attribute_value(value)
        for index, role in enumerate(spec.role_instructions):
            trajectory.attrs[f"role_instruction_panda_{index}"] = role

        actions = np.asarray(episode.actions, dtype=np.float32)
        for index, camera in enumerate(spec.local_cameras):
            _dataset(
                trajectory,
                f"actions/panda-{index}",
                actions[:, index * 7 : (index + 1) * 7],
            )
            _dataset(
                trajectory,
                f"obs/agent/panda-{index}/qpos",
                np.asarray(episode.proprio[index], dtype=np.float32),
            )
            _dataset(
                trajectory,
                f"obs/sensor_data/{camera}/rgb",
                np.asarray(episode.local_images[index], dtype=np.uint8),
                images=True,
            )
        _dataset(
            trajectory,
            f"obs/sensor_data/{spec.global_camera}/rgb",
            np.asarray(episode.global_images, dtype=np.uint8),
            images=True,
        )
        _dataset(trajectory, "sim/states", np.asarray(episode.sim_states, dtype=np.float64))
        trajectory.create_dataset(
            "sim/final_state", data=np.asarray(final_sim_state, dtype=np.float64)
        )
        trajectory.create_dataset(
            "timestamps",
            data=np.arange(len(episode), dtype=np.float64) / spec.control_frequency_hz,
        )
        trajectory.create_dataset("rewards", data=np.asarray(episode.rewards, dtype=np.float32))
        trajectory.create_dataset("dones", data=np.asarray(episode.dones, dtype=np.bool_))
        trajectory.create_dataset("successes", data=np.asarray(episode.successes, dtype=np.bool_))
        trajectory.create_dataset("task_stage", data=np.asarray(episode.stages, dtype=np.int16))
        trajectory.create_dataset("active_arm", data=np.asarray(episode.active_arms, dtype=np.int8))
        trajectory.create_dataset(
            "control_mode",
            data=np.asarray(episode.control_modes, dtype=h5py.string_dtype("utf-8")),
        )
        trajectory.create_dataset("grasp_flags", data=np.asarray(episode.grasp_flags, dtype=np.bool_))
        trajectory.create_dataset(
            "wall_timestamps", data=np.asarray(episode.wall_timestamps, dtype=np.float64)
        )
        trajectory.create_dataset(
            "capture_reason", data=np.asarray(episode.capture_reasons, dtype=np.uint16)
        )

        trajectory.attrs.modify("complete", True)
        handle.move(temporary_name, name)
        handle.flush()
    return name


def audit_multiarm_dataset(
    path: str | Path,
    *,
    expected_agents: int | None = None,
    require_success: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    report: dict[str, Any] = {
        "path": str(path.resolve()),
        "schema_version": None,
        "num_agents": None,
        "trajectories": [],
        "total_steps": 0,
        "successful_trajectories": 0,
    }
    with h5py.File(path, "r") as handle:
        report["schema_version"] = str(handle.attrs.get("schema_version", "missing"))
        num_agents = int(handle.attrs["num_agents"])
        report["num_agents"] = num_agents
        if expected_agents is not None and num_agents != expected_agents:
            raise ValueError(f"expected {expected_agents} agents, found {num_agents}")
        names = _trajectory_names(handle)
        if not names:
            raise ValueError("Dataset contains no complete trajectories")
        for name in names:
            trajectory = handle[name]
            if not bool(trajectory.attrs.get("complete", False)):
                raise ValueError(f"{name} is incomplete")
            success = bool(trajectory.attrs.get("success", False))
            if require_success and not success:
                raise ValueError(f"{name} is not successful")
            camera_names = json.loads(str(trajectory.attrs["local_cameras_json"]))
            global_camera = str(trajectory.attrs["global_camera"])
            sequences = [trajectory["sim/states"], trajectory["grasp_flags"]]
            for index in range(num_agents):
                action = trajectory[f"actions/panda-{index}"]
                proprio = trajectory[f"obs/agent/panda-{index}/qpos"]
                if action.shape[1:] != (7,):
                    raise ValueError(f"{name} panda-{index} action is not [T,7]")
                if proprio.shape[1:] != (9,):
                    raise ValueError(f"{name} panda-{index} proprio is not [T,9]")
                sequences.extend([action, proprio, trajectory[f"obs/sensor_data/{camera_names[index]}/rgb"]])
            sequences.append(trajectory[f"obs/sensor_data/{global_camera}/rgb"])
            lengths = [int(sequence.shape[0]) for sequence in sequences]
            if len(set(lengths)) != 1:
                raise ValueError(f"{name} has misaligned lengths: {lengths}")
            if trajectory["grasp_flags"].shape[1:] != (num_agents,):
                raise ValueError(f"{name} grasp_flags is not [T,{num_agents}]")
            for sequence in sequences:
                if not np.isfinite(sequence[:]).all():
                    raise ValueError(f"{name} contains non-finite values")
            steps = lengths[0]
            report["trajectories"].append(
                {"name": name, "steps": steps, "success": success, "source": str(trajectory.attrs["source"])}
            )
            report["total_steps"] += steps
            report["successful_trajectories"] += int(success)
    return report


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return np.asarray(image)
    return cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_AREA)


def montage_frame(global_image: np.ndarray, local_images: Sequence[np.ndarray]) -> np.ndarray:
    """Lay out a global view on the left and 1/2/4 wrist views on the right."""

    global_image = np.asarray(global_image, dtype=np.uint8)
    height, width = global_image.shape[:2]
    if len(local_images) == 1:
        right = _resize_rgb(local_images[0], width, height)
    elif len(local_images) == 2:
        half_height = height // 2
        rows = [_resize_rgb(image, width, half_height) for image in local_images]
        right = np.concatenate(rows, axis=0)
        right = _resize_rgb(right, width, height)
    elif len(local_images) == 4:
        half_width, half_height = width // 2, height // 2
        tiles = [_resize_rgb(image, half_width, half_height) for image in local_images]
        right = np.concatenate(
            [np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0
        )
        right = _resize_rgb(right, width, height)
    else:
        raise ValueError("montage supports exactly 1, 2, or 4 local cameras")
    return np.concatenate([global_image, right], axis=1)


def buffered_episode_frame(episode: MultiArmEpisodeBuffer, index: int) -> np.ndarray:
    index = min(max(int(index), 0), len(episode) - 1)
    return montage_frame(
        episode.global_images[index],
        [images[index] for images in episode.local_images],
    )


def combined_episode_frame(path: str | Path, name: str, index: int) -> tuple[np.ndarray, int]:
    with h5py.File(path, "r") as handle:
        if name not in _trajectory_names(handle):
            raise KeyError(name)
        trajectory = handle[name]
        total = int(trajectory["actions/panda-0"].shape[0])
        index = min(max(int(index), 0), total - 1)
        global_camera = str(trajectory.attrs["global_camera"])
        local_cameras = json.loads(str(trajectory.attrs["local_cameras_json"]))
        global_image = trajectory[f"obs/sensor_data/{global_camera}/rgb"][index]
        locals_ = [trajectory[f"obs/sensor_data/{camera}/rgb"][index] for camera in local_cameras]
        return montage_frame(global_image, locals_), total


def list_episodes(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with h5py.File(path, "r") as handle:
        return [
            {
                "name": name,
                "steps": int(handle[name]["actions/panda-0"].shape[0]),
                "success": bool(handle[name].attrs.get("success", False)),
                "source": str(handle[name].attrs.get("source", "unknown")),
                "created_at": str(handle[name].attrs.get("created_at", "")),
                "spawn_side": str(handle[name].attrs.get("spawn_side", "")),
            }
            for name in reversed(_trajectory_names(handle))
        ]


def trash_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}.trash{path.suffix or '.h5'}")


def recoverable_delete(path: str | Path, name: str) -> str:
    path = Path(path)
    destination = trash_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trash_name = f"deleted_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    with h5py.File(path, "a") as source, h5py.File(destination, "a") as trash:
        if name not in _trajectory_names(source):
            raise KeyError(name)
        source.copy(name, trash, name=trash_name)
        trash[trash_name].attrs["original_name"] = name
        trash[trash_name].attrs["deleted_at"] = datetime.now(timezone.utc).isoformat()
        trash.flush()
        del source[name]
        source.flush()
    return trash_name


def list_trash(path: str | Path) -> list[dict[str, Any]]:
    destination = trash_path(path)
    if not destination.exists():
        return []
    with h5py.File(destination, "r") as handle:
        return [
            {
                "trash_name": name,
                "original_name": str(group.attrs.get("original_name", "")),
                "deleted_at": str(group.attrs.get("deleted_at", "")),
                "steps": int(group["actions/panda-0"].shape[0]),
            }
            for name, group in reversed(list(handle.items()))
        ]


def restore_episode(path: str | Path, trash_name: str) -> str:
    path = Path(path)
    destination = trash_path(path)
    with h5py.File(destination, "a") as trash, h5py.File(path, "a") as target:
        if trash_name not in trash:
            raise KeyError(trash_name)
        preferred = str(trash[trash_name].attrs.get("original_name", ""))
        name = preferred if preferred and preferred not in target else _next_trajectory_name(target)
        trash.copy(trash_name, target, name=name)
        target[name].attrs["restored_at"] = datetime.now(timezone.utc).isoformat()
        target.flush()
        del trash[trash_name]
        trash.flush()
    return name
