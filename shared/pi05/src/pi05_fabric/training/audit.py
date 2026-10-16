"""Strict completion checks for persisted training checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pi05_fabric.training.checkpoint import restore_training_state


def audit_training_checkpoint(
    path: Path,
    *,
    expected_stage: str,
    expected_step: int,
    verify_model: bool = True,
    expected_parent_model_sha256: str | None = None,
) -> dict[str, object]:
    """Validate checkpoint files, metadata, hashes, and optional lineage."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    state_path = path / "state.msgpack"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing checkpoint manifest: {manifest_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"missing checkpoint state: {state_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 3:
        raise ValueError("checkpoint schema mismatch")
    if manifest.get("stage") != expected_stage:
        raise ValueError("checkpoint stage mismatch")
    if int(manifest.get("step", -1)) != expected_step:
        raise ValueError("checkpoint step mismatch")
    if int(manifest.get("schedule_step", -1)) != expected_step:
        raise ValueError("checkpoint schedule step mismatch")

    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    if state_sha256 != manifest.get("state_sha256"):
        raise ValueError("checkpoint state hash mismatch")
    if expected_parent_model_sha256 is not None:
        if manifest.get("parent_model_sha256") != expected_parent_model_sha256:
            raise ValueError("checkpoint parent model hash mismatch")

    if verify_model:
        snapshot = restore_training_state(path)
        if snapshot.stage.value != expected_stage or snapshot.step != expected_step:
            raise ValueError("restored checkpoint metadata mismatch")
    return manifest
