#!/usr/bin/env python3
"""Isolated MuJoCo coordinator for four-arm PI0.5 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import pickle
import socket
import struct

import numpy as np


_HEADER = struct.Struct("!Q")
GLOBAL_CAMERA = "agentview"
LOCAL_CAMERAS = tuple(f"robot{role}_eye_in_hand" for role in range(4))


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise EOFError("four-arm environment connection closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _recv(connection: socket.socket):
    size = _HEADER.unpack(_recv_exact(connection, _HEADER.size))[0]
    return pickle.loads(_recv_exact(connection, size))


def _send(connection: socket.socket, payload) -> None:
    data = pickle.dumps(payload, protocol=4)
    connection.sendall(_HEADER.pack(len(data)))
    connection.sendall(data)


def _rgb(observation: dict, camera: str) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(observation[f"{camera}_image"]))


def _proprio(observation: dict, role: int) -> np.ndarray:
    return np.concatenate(
        [observation[f"robot{role}_joint_pos"], observation[f"robot{role}_gripper_qpos"]]
    ).astype(np.float32)


def _observation(observation: dict) -> dict:
    return {
        "global_rgb": _rgb(observation, GLOBAL_CAMERA),
        "wrist_rgb": tuple(_rgb(observation, camera) for camera in LOCAL_CAMERAS),
        "qpos": tuple(_proprio(observation, role) for role in range(4)),
    }


def _initial_condition(observation: dict, task_status: dict) -> dict:
    digest = hashlib.sha256()
    digest.update(_rgb(observation, GLOBAL_CAMERA).tobytes())
    qpos_digest = hashlib.sha256()
    for role in range(4):
        qpos_digest.update(_proprio(observation, role).tobytes())
    return {
        "initial_global_rgb_sha256": digest.hexdigest(),
        "initial_qpos_sha256": qpos_digest.hexdigest(),
        "task_status": task_status,
    }


def _make_environment(task: str, *, seed: int, horizon: int, image_size: int):
    if task == "frame_insertion":
        from multiarm_sim.envs.frame_insertion import make_frame_insertion_env

        return make_frame_insertion_env(
            global_image_size=image_size,
            wrist_image_size=image_size,
            horizon=horizon,
            seed=seed,
        )
    if task == "arch_assembly":
        from multiarm_sim.envs.arch_assembly import make_arch_assembly_env

        return make_arch_assembly_env(
            global_image_size=image_size,
            wrist_image_size=image_size,
            horizon=horizon,
            seed=seed,
        )
    raise ValueError(f"unsupported task: {task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--task", choices=("frame_insertion", "arch_assembly"), required=True)
    parser.add_argument("--maximum-steps", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(1)
    connection = None
    environment = None
    observation = None
    elapsed_steps = 0
    try:
        connection, _ = server.accept()
        _send(connection, {"type": "ready", "action_dim": 28, "agent_count": 4})
        while True:
            request = _recv(connection)
            command = request["command"]
            if command == "close":
                _send(connection, {"type": "closed"})
                break
            if command == "reset":
                if environment is not None:
                    environment.close()
                seed = int(request["seed"])
                np.random.seed(seed)
                environment = _make_environment(
                    args.task,
                    seed=seed,
                    horizon=args.maximum_steps,
                    image_size=args.image_size,
                )
                observation = environment.reset()
                elapsed_steps = 0
                status = environment.task_status()
                _send(
                    connection,
                    {
                        "type": "observation",
                        "observation": _observation(observation),
                        "task_status": status,
                        "success": bool(status["success"]),
                        "elapsed_steps": 0,
                        "initial_condition": _initial_condition(observation, status),
                    },
                )
                continue
            if command != "step_chunk" or environment is None or observation is None:
                raise ValueError(f"invalid environment command: {command}")
            actions = np.asarray(request["actions"], dtype=np.float32)
            if actions.ndim != 3 or actions.shape[0] != 4 or actions.shape[2] != 7:
                raise ValueError(f"expected [4, horizon, 7] actions, got {actions.shape}")
            if not np.isfinite(actions).all():
                raise FloatingPointError("received non-finite four-arm actions")
            executed = 0
            done = False
            for index in range(actions.shape[1]):
                observation, _, _, _ = environment.step(actions[:, index].reshape(-1))
                elapsed_steps += 1
                executed += 1
                status = environment.task_status()
                done = bool(status["success"] or elapsed_steps >= args.maximum_steps)
                if done:
                    break
            _send(
                connection,
                {
                    "type": "observation",
                    "observation": _observation(observation),
                    "task_status": status,
                    "success": bool(status["success"]),
                    "truncated": elapsed_steps >= args.maximum_steps,
                    "executed_steps": executed,
                    "elapsed_steps": elapsed_steps,
                },
            )
    finally:
        if connection is not None:
            connection.close()
        if environment is not None:
            environment.close()
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
