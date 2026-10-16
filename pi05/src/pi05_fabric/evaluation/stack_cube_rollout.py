"""Pure adapters for strict three-agent RoboFactory rollouts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


GLOBAL_CAMERA = "head_camera_global"
WRIST_CAMERAS = tuple(f"wrist_camera_agent{role}" for role in range(3))


def _remove_environment_batch(value, *, expected_rank: int):
    array = np.asarray(value)
    if array.ndim == expected_rank + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != expected_rank:
        raise ValueError(f"environment value must have rank {expected_rank}")
    return array


def extract_role_observation(payload: Mapping, *, role_index: int) -> dict:
    if not 0 <= role_index < 3:
        raise ValueError("role_index must identify one of three agents")
    images = payload.get("images")
    qpos = payload.get("qpos")
    if not isinstance(images, Mapping) or not isinstance(qpos, (list, tuple)):
        raise ValueError("environment payload is missing images or role qpos")
    required = (GLOBAL_CAMERA, *WRIST_CAMERAS)
    missing = [name for name in required if name not in images]
    if missing or len(qpos) != 3:
        raise ValueError(f"incomplete three-agent observation: missing={missing}")
    global_rgb = _remove_environment_batch(images[GLOBAL_CAMERA], expected_rank=3)
    wrist_rgb = _remove_environment_batch(
        images[WRIST_CAMERAS[role_index]],
        expected_rank=3,
    )
    state = _remove_environment_batch(qpos[role_index], expected_rank=1)
    if global_rgb.shape[-1] != 3 or wrist_rgb.shape[-1] != 3:
        raise ValueError("RGB observations must have three channels")
    if state.shape != (9,):
        raise ValueError("Stack Cube role state must have shape (9,)")
    return {
        "role_index": role_index,
        "global_rgb": global_rgb,
        "wrist_rgb": wrist_rgb,
        "state": state.astype(np.float32),
    }


def rollout_record(*, seed: int, planning_rounds: int, response: Mapping) -> dict:
    diagnostics = dict(response.get("task_diagnostics") or {})
    required = (
        "strict_success",
        "official_success",
        "cubeB_on_cubeA",
        "cubeC_on_cubeB",
        "cubeB_in_goal",
        "cubeC_in_goal",
        "cubeA_grasped_by_agent0",
        "cubeB_grasped_by_agent1",
        "cubeC_grasped_by_agent2",
    )
    missing = [key for key in required if key not in diagnostics]
    if missing:
        raise ValueError(f"strict Stack Cube diagnostics are missing: {missing}")
    return {
        "seed": int(seed),
        "strict_success": bool(diagnostics["strict_success"]),
        "official_success": bool(diagnostics["official_success"]),
        "planning_rounds": int(planning_rounds),
        "elapsed_steps": int(response["elapsed_steps"]),
        "task_diagnostics": diagnostics,
        "initial_condition": dict(response.get("initial_condition") or {}),
        "timeout": bool(response.get("truncated", False)),
        "terminated": bool(response.get("terminated", False)),
    }
