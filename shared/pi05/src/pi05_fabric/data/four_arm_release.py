"""Integrity checks for formal four-arm HDF5 source releases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np


class SourceReleaseError(ValueError):
    """Raised when a source dataset is not eligible for formal training."""


@dataclass(frozen=True)
class SourceReleaseReport:
    task_id: str
    path: Path
    num_agents: int
    episodes: int
    successes: int
    unique_seed_count: int
    training_ready: bool
    total_steps: int


def _decoded_json(value: object, *, attribute: str) -> list[str]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        result = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise SourceReleaseError(f"invalid {attribute}: {value!r}") from error
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise SourceReleaseError(f"{attribute} must be a JSON string list")
    return result


def _task_id(handle: h5py.File, trajectories: list[str]) -> str:
    environment = str(handle.attrs.get("environment", ""))
    if "FourArmFrameInsertion" in environment:
        return "frame_insertion"
    first = handle[trajectories[0]]
    task_id = str(first.attrs.get("task_id", ""))
    if task_id == "f2_arch_assembly" or "FourArmArchAssembly" in environment:
        return "f2_arch_assembly"
    raise SourceReleaseError(f"unsupported four-arm environment: {environment!r}")


def _formal_release_metadata_confirms(path: Path, expected_episodes: int) -> bool:
    audit_path = path.parent / "audit.json"
    manifest_path = path.parent / "manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    checks = audit.get("checks", {})
    hdf5 = manifest.get("hdf5", {})
    hdf5_path = hdf5.get("path") if isinstance(hdf5, dict) else str(hdf5)
    return bool(
        audit.get("trajectory_count") == expected_episodes
        and audit.get("successful_trajectories") == expected_episodes
        and checks.get("all_trajectories_successful") is True
        and checks.get("episode_ids_contiguous") is True
        and checks.get("seeds_contiguous") is True
        and hdf5_path == path.name
        and manifest.get("version") == "1.0.0"
    )


def _check_role_arrays(trajectory: h5py.Group, steps: int) -> None:
    for role in range(4):
        action_key = f"actions/panda-{role}"
        qpos_candidates = (
            f"obs/agent/panda-{role}/qpos",
            f"obs/robot_state/panda-{role}_joint_pos",
        )
        qpos_key = next((key for key in qpos_candidates if key in trajectory), None)
        if action_key not in trajectory or qpos_key is None:
            raise SourceReleaseError(f"missing role-{role} action or qpos stream")
        actions = trajectory[action_key]
        qpos = trajectory[qpos_key]
        if actions.shape != (steps, 7):
            raise SourceReleaseError(f"{action_key} has shape {actions.shape}, expected {(steps, 7)}")
        if qpos.shape != (steps, 9):
            raise SourceReleaseError(f"{qpos_key} has shape {qpos.shape}, expected {(steps, 9)}")
        if not np.isfinite(actions[:]).all() or not np.isfinite(qpos[:]).all():
            raise SourceReleaseError(f"role-{role} contains non-finite values")


def _check_cameras(trajectory: h5py.Group, steps: int) -> None:
    global_cameras = _decoded_json(
        trajectory.attrs.get("global_cameras_json", '["agentview"]'),
        attribute="global_cameras_json",
    )
    local_cameras = _decoded_json(
        trajectory.attrs.get("local_cameras_json", "[]"),
        attribute="local_cameras_json",
    )
    if "agentview" not in global_cameras or len(local_cameras) != 4:
        raise SourceReleaseError("release must contain agentview and four wrist cameras")
    for camera in ("agentview", *local_cameras):
        key = f"obs/sensor_data/{camera}/rgb"
        if key not in trajectory or trajectory[key].shape[0] != steps:
            raise SourceReleaseError(f"camera stream is missing or misaligned: {key}")


def audit_source_release(
    path: str | Path, *, expected_episodes: int = 50
) -> SourceReleaseReport:
    """Validate a success-only, four-agent formal source release."""

    path = Path(path)
    if not path.is_file():
        raise SourceReleaseError(f"source release does not exist: {path}")
    try:
        with h5py.File(path, "r") as handle:
            num_agents = int(handle.attrs.get("num_agents", -1))
            declared_training_ready = handle.attrs.get("training_ready")
            declared_preview_only = handle.attrs.get("preview_only")
            trajectories = sorted(key for key in handle if key.startswith("trajectory_"))
            if num_agents != 4:
                raise SourceReleaseError(f"expected four agents, found {num_agents}")
            if len(trajectories) != expected_episodes:
                raise SourceReleaseError(
                    f"expected {expected_episodes} episodes, found {len(trajectories)}"
                )
            metadata_ready = _formal_release_metadata_confirms(path.resolve(), expected_episodes)
            training_ready = (
                bool(declared_training_ready)
                if declared_training_ready is not None
                else metadata_ready
            )
            preview_only = (
                bool(declared_preview_only)
                if declared_preview_only is not None
                else not metadata_ready
            )
            if not training_ready or preview_only:
                raise SourceReleaseError("dataset must declare training_ready=true and preview_only=false")

            task_id = _task_id(handle, trajectories)
            successes = 0
            seeds: set[int] = set()
            total_steps = 0
            for index, name in enumerate(trajectories):
                if name != f"trajectory_{index:06d}":
                    raise SourceReleaseError("trajectory identifiers must be contiguous")
                trajectory = handle[name]
                success = bool(trajectory.attrs.get("success", False))
                complete = bool(trajectory.attrs.get("complete", False))
                if not success or not complete:
                    raise SourceReleaseError(f"{name} is not a complete success")
                seed = int(trajectory.attrs["seed"])
                if seed in seeds:
                    raise SourceReleaseError(f"duplicate seed: {seed}")
                seeds.add(seed)
                steps = int(trajectory["actions/panda-0"].shape[0])
                if steps <= 0:
                    raise SourceReleaseError(f"{name} is empty")
                _check_role_arrays(trajectory, steps)
                _check_cameras(trajectory, steps)
                successes += 1
                total_steps += steps
    except OSError as error:
        raise SourceReleaseError(f"cannot read source release: {path}") from error

    return SourceReleaseReport(
        task_id=task_id,
        path=path.resolve(),
        num_agents=num_agents,
        episodes=len(trajectories),
        successes=successes,
        unique_seed_count=len(seeds),
        training_ready=training_ready,
        total_steps=total_steps,
    )


__all__ = ("SourceReleaseError", "SourceReleaseReport", "audit_source_release")
