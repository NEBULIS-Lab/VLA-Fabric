"""Length-prefixed pickle client matching the audited RoboFactory server."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import socket
import struct
import tempfile


_HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def short_environment_socket_path(output: Path) -> Path:
    digest = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"pi05_sc_{digest}.sock"


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("RoboFactory IPC connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_pickle(connection: socket.socket, payload) -> None:
    data = pickle.dumps(payload, protocol=4)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("RoboFactory IPC payload exceeds the safety limit")
    connection.sendall(_HEADER.pack(len(data)))
    connection.sendall(data)


def receive_pickle(connection: socket.socket):
    size = _HEADER.unpack(_receive_exact(connection, _HEADER.size))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("RoboFactory IPC payload exceeds the safety limit")
    return pickle.loads(_receive_exact(connection, size))
