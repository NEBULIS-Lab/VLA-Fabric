"""Dependency-light local IPC for the RoboTwin and OpenPI runtimes."""

from __future__ import annotations

from io import BytesIO
import socket
import struct
from typing import Mapping

import numpy as np

from pi05_fabric.evaluation.robotwin2_adapter import ACTION_HORIZON
from pi05_fabric.evaluation.robotwin2_adapter import POLICY_ACTION_DIM
from pi05_fabric.evaluation.robotwin2_adapter import validate_policy_payload


MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct("!Q")


def paired_flow_seed(condition_id: int, planning_call: int) -> int:
    """Derive a stable uint32 seed for paired flow noise at one planning call."""
    if condition_id < 0 or planning_call < 0:
        raise ValueError("condition_id and planning_call must be non-negative")
    return int((condition_id * 1_000_003 + planning_call * 97) % (2**32))


def encode_npz_message(fields: Mapping[str, object]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **{key: np.asarray(value) for key, value in fields.items()})
    payload = stream.getvalue()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"IPC payload exceeds {MAX_MESSAGE_BYTES} bytes")
    return _HEADER.pack(len(payload)) + payload


def decode_npz_message(payload: bytes) -> dict[str, np.ndarray]:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"IPC payload exceeds {MAX_MESSAGE_BYTES} bytes")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("IPC peer closed before the message completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(connection: socket.socket) -> dict[str, np.ndarray]:
    (size,) = _HEADER.unpack(_recv_exact(connection, _HEADER.size))
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"declared IPC payload exceeds {MAX_MESSAGE_BYTES} bytes")
    return decode_npz_message(_recv_exact(connection, size))


def send_message(connection: socket.socket, fields: Mapping[str, object]) -> None:
    connection.sendall(encode_npz_message(fields))


def deterministic_smoke_response(request: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Produce a deterministic, non-executed action chunk for bridge testing."""
    validated = validate_policy_payload(request)
    request_id = str(validated["request_id"].item())
    condition_id = int(validated["condition_id"].item())
    planning_call = int(validated["planning_call"].item())
    seed = paired_flow_seed(condition_id, planning_call)
    generator = np.random.default_rng(seed)
    actions = generator.normal(0.0, 0.01, size=(ACTION_HORIZON, POLICY_ACTION_DIM)).astype(np.float32)
    identity_r6 = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    actions[:, 3:9] = identity_r6
    actions[:, 13:19] = identity_r6
    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "status": np.asarray("ok"),
        "request_id": np.asarray(request_id),
        "flow_seed": np.asarray(seed, dtype=np.uint32),
        "actions_r6": actions,
    }


def validate_policy_response(response: Mapping[str, object], *, request_id: str) -> np.ndarray:
    if int(np.asarray(response["schema_version"]).item()) != 1:
        raise ValueError("unsupported policy response schema_version")
    if str(np.asarray(response["status"]).item()) != "ok":
        error = str(np.asarray(response.get("error", "unknown IPC error")).item())
        raise RuntimeError(error)
    if str(np.asarray(response["request_id"]).item()) != request_id:
        raise ValueError("policy response request_id does not match request")
    actions = np.asarray(response["actions_r6"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, POLICY_ACTION_DIM):
        raise ValueError(f"actions_r6 must have shape {(ACTION_HORIZON, POLICY_ACTION_DIM)}")
    if not np.isfinite(actions).all():
        raise ValueError("actions_r6 contains non-finite values")
    return actions
