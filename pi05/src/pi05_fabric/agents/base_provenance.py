"""Content fingerprints for immutable base checkpoints; no model/device imports."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

from pi05_fabric.provenance import _sha256 as file_sha256


def base_tree_sha256(files: Mapping[str, str]) -> str:
    """Match the file-map encoding used by the established Mic provenance record."""
    return hashlib.sha256(json.dumps(dict(files), sort_keys=True).encode("utf-8")).hexdigest()


def fingerprint_base_checkpoint(path: str | Path) -> dict:
    """Re-read every file, binding relative names and contents, not paths/mtimes."""
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"base checkpoint must be a directory: {root}")
    files = {}
    for entry in sorted(root.rglob("*")):
        if entry.is_dir():
            # Path.rglob does not recurse through directory symlinks.
            if entry.is_symlink():
                raise ValueError(f"base checkpoint contains an untraversed directory symlink: {entry}")
            continue
        if not entry.is_file():
            raise ValueError(f"base checkpoint contains a missing or non-regular file: {entry}")
        files[entry.relative_to(root).as_posix()] = file_sha256(entry)
    if not files:
        raise ValueError(f"base checkpoint contains no files: {root}")
    return {"resolved_path": str(root), "files": files, "tree_sha256": base_tree_sha256(files)}


def validate_base_fingerprint(protocol: Mapping | None, actual_tree_sha256: str) -> None:
    """Fail closed for Direct snapshots without an explicit matching base digest."""
    initialization = protocol.get("initialization") if isinstance(protocol, Mapping) else None
    expected = initialization.get("base_tree_sha256") if isinstance(initialization, Mapping) else None
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Full-direct base tree hash is missing or invalid in checkpoint protocol")
    if expected != actual_tree_sha256:
        raise ValueError(
            f"Full-direct base tree hash mismatch: expected {expected}, found {actual_tree_sha256}"
        )
