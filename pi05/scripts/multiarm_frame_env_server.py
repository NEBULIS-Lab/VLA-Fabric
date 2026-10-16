#!/usr/bin/env python3
"""Isolated MuJoCo coordinator for three- and four-arm frame evaluation."""

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


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise EOFError("multi-arm environment connection closed")
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


def _observation(observation: dict, *, agent_count: int) -> dict:
    cameras = tuple(f"robot{role}_eye_in_hand" for role in range(agent_count))
    return {
        "global_rgb": _rgb(observation, GLOBAL_CAMERA),
        "wrist_rgb": tuple(_rgb(observation, camera) for camera in cameras),
        "qpos": tuple(_proprio(observation, role) for role in range(agent_count)),
    }


def _initial_condition(observation: dict, task_status: dict, *, agent_count: int) -> dict:
    image_digest = hashlib.sha256(_rgb(observation, GLOBAL_CAMERA).tobytes())
    qpos_digest = hashlib.sha256()
    for role in range(agent_count):
        qpos_digest.update(_proprio(observation, role).tobytes())
    return {
        "initial_global_rgb_sha256": image_digest.hexdigest(),
        "initial_qpos_sha256": qpos_digest.hexdigest(),
        "task_status": task_status,
    }


def _make_environment(task: str, *, seed: int, horizon: int, image_size: int):
    if task == "frame4_insertion":
        from multiarm_sim.envs.frame_insertion import make_frame_insertion_env

        return make_frame_insertion_env(
            global_image_size=image_size,
            wrist_image_size=image_size,
            horizon=horizon,
            seed=seed,
        )
    if task == "frame3_triangle_insertion":
        from multiarm_sim.envs.triangle_frame_insertion import (
            make_triangle_frame_insertion_env,
        )

        return make_triangle_frame_insertion_env(
            global_image_size=image_size,
            wrist_image_size=image_size,
            horizon=horizon,
            seed=seed,
            spawn_side="left" if seed % 2 == 0 else "right",
        )
    raise ValueError(f"unsupported task: {task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("frame4_insertion", "frame3_triangle_insertion"),
        required=True,
    )
    parser.add_argument("--agent-count", type=int, choices=(3, 4), required=True)
    parser.add_argument("--maximum-steps", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = 4 if args.task == "frame4_insertion" else 3
    if args.agent_count != expected:
        raise ValueError("task and agent count are inconsistent")
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
        _send(
            connection,
            {
                "type": "ready",
                "action_dim": 7 * args.agent_count,
                "agent_count": args.agent_count,
            },
        )
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
                        "observation": _observation(
                            observation, agent_count=args.agent_count
                        ),
                        "task_status": status,
                        "success": bool(status["success"]),
                        "elapsed_steps": 0,
                        "initial_condition": _initial_condition(
                            observation,
                            status,
                            agent_count=args.agent_count,
                        ),
                    },
                )
                continue
            if command != "step_chunk" or environment is None or observation is None:
                raise ValueError(f"invalid environment command: {command}")
            actions = np.asarray(request["actions"], dtype=np.float32)
            if (
                actions.ndim != 3
                or actions.shape[0] != args.agent_count
                or actions.shape[2] != 7
            ):
                raise ValueError(
                    f"expected [{args.agent_count}, horizon, 7] actions, got "
                    f"{actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise FloatingPointError("received non-finite multi-arm actions")
            executed = 0
            for index in range(actions.shape[1]):
                observation, _, _, _ = environment.step(actions[:, index].reshape(-1))
                elapsed_steps += 1
                executed += 1
                status = environment.task_status()
                if status["success"] or elapsed_steps >= args.maximum_steps:
                    break
            _send(
                connection,
                {
                    "type": "observation",
                    "observation": _observation(
                        observation, agent_count=args.agent_count
                    ),
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
