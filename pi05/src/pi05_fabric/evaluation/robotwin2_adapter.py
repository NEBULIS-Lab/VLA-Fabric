"""RoboTwin2 observation and end-effector action adapters."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


POLICY_ACTION_DIM = 20
ROBOTWIN_EE_ACTION_DIM = 16
ACTION_HORIZON = 50


def _finite_vector(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _rgb_image(value: Any, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 pixels, got {image.dtype}")
    return np.ascontiguousarray(image)


def quaternion_wxyz_to_r6(quaternion: Any) -> np.ndarray:
    """Convert a RoboTwin scalar-first quaternion to matrix-column R6."""
    quat = _finite_vector(quaternion, shape=(4,), name="quaternion_wxyz")
    if np.linalg.norm(quat) < 1e-8:
        raise ValueError("quaternion_wxyz has zero norm")
    matrix = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def r6_to_quaternion_wxyz(rotation_r6: Any) -> np.ndarray:
    """Convert matrix-column R6 to a normalized scalar-first quaternion."""
    r6 = _finite_vector(rotation_r6, shape=(6,), name="rotation_r6")
    first = r6[:3]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError("rotation_r6 first column is degenerate")
    x_axis = first / first_norm
    second = r6[3:] - np.dot(x_axis, r6[3:]) * x_axis
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        raise ValueError("rotation_r6 columns are collinear")
    y_axis = second / second_norm
    z_axis = np.cross(x_axis, y_axis)
    matrix = np.stack((x_axis, y_axis, z_axis), axis=1)
    return Rotation.from_matrix(matrix).as_quat(scalar_first=True).astype(np.float32)


def observation_to_policy_payload(
    observation: Mapping[str, Any],
    *,
    instruction: str,
    condition_id: int,
    planning_call: int,
    request_id: str,
) -> dict[str, np.ndarray]:
    """Map one RoboTwin observation to the process-boundary policy schema."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")

    camera = observation["observation"]
    endpose = observation["endpose"]
    left_pose = _finite_vector(endpose["left_endpose"], shape=(7,), name="left_endpose")
    right_pose = _finite_vector(endpose["right_endpose"], shape=(7,), name="right_endpose")
    left_gripper = _finite_vector([endpose["left_gripper"]], shape=(1,), name="left_gripper")
    right_gripper = _finite_vector([endpose["right_gripper"]], shape=(1,), name="right_gripper")

    left_state = np.concatenate(
        (left_pose[:3], quaternion_wxyz_to_r6(left_pose[3:]), left_gripper)
    ).astype(np.float32)
    right_state = np.concatenate(
        (right_pose[:3], quaternion_wxyz_to_r6(right_pose[3:]), right_gripper)
    ).astype(np.float32)

    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "kind": np.asarray("inference"),
        "request_id": np.asarray(str(request_id)),
        "condition_id": np.asarray(condition_id, dtype=np.int64),
        "planning_call": np.asarray(planning_call, dtype=np.int64),
        "instruction": np.asarray(instruction),
        "global_image": _rgb_image(camera["head_camera"]["rgb"], name="global_image"),
        "left_wrist_image": _rgb_image(camera["left_camera"]["rgb"], name="left_wrist_image"),
        "right_wrist_image": _rgb_image(camera["right_camera"]["rgb"], name="right_wrist_image"),
        "eef_state": np.concatenate((left_state, right_state)).astype(np.float32),
    }


def validate_policy_payload(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "schema_version",
        "kind",
        "request_id",
        "condition_id",
        "planning_call",
        "instruction",
        "global_image",
        "left_wrist_image",
        "right_wrist_image",
        "eef_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"policy payload is missing fields: {missing}")
    if int(np.asarray(payload["schema_version"]).item()) != 1:
        raise ValueError("unsupported policy payload schema_version")
    if str(np.asarray(payload["kind"]).item()) != "inference":
        raise ValueError("policy payload kind must be 'inference'")
    if not str(np.asarray(payload["request_id"]).item()):
        raise ValueError("request_id must be non-empty")
    if not str(np.asarray(payload["instruction"]).item()).strip():
        raise ValueError("instruction must be non-empty")
    for key in ("global_image", "left_wrist_image", "right_wrist_image"):
        _rgb_image(payload[key], name=key)
    _finite_vector(payload["eef_state"], shape=(POLICY_ACTION_DIM,), name="eef_state")
    return {key: np.asarray(value) for key, value in payload.items()}


def policy_r6_chunk_to_robotwin_ee(actions: Any) -> np.ndarray:
    """Convert a policy action chunk `(H, 20)` to RoboTwin `(H, 18)` EE actions."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != POLICY_ACTION_DIM:
        raise ValueError(f"actions must have shape (H, {POLICY_ACTION_DIM}), got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("actions contain non-finite values")

    converted = np.empty((chunk.shape[0], ROBOTWIN_EE_ACTION_DIM), dtype=np.float32)
    converted[:, 0:3] = chunk[:, 0:3]
    converted[:, 3:7] = np.stack([r6_to_quaternion_wxyz(row[3:9]) for row in chunk])
    converted[:, 7] = chunk[:, 9]
    converted[:, 8:11] = chunk[:, 10:13]
    converted[:, 11:15] = np.stack([r6_to_quaternion_wxyz(row[13:19]) for row in chunk])
    converted[:, 15] = chunk[:, 19]
    return converted
