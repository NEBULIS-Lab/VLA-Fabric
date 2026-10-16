"""Strict completion checks for persisted training checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pi05_fabric.training.checkpoint import restore_training_state
from pi05_fabric.training.stages import StageName


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
    if expected_stage == StageName.PI_NATIVE_V2_FULL_DIRECT.value:
        protocol = manifest.get("protocol")
        initialization = protocol.get("initialization") if isinstance(protocol, dict) else None
        if (manifest.get("parent_model_sha256", "missing") is not None
                or not isinstance(initialization, dict)
                or initialization.get("kind") != "base_direct"
                or initialization.get("parent_model_sha256", "missing") is not None):
            raise ValueError("Full-direct checkpoint requires base_direct initialization without a parent hash")
        base_checkpoint = initialization.get("base_checkpoint")
        if not isinstance(base_checkpoint, str) or not Path(base_checkpoint).is_absolute():
            raise ValueError("Full-direct checkpoint requires an absolute base checkpoint path")
        if "continuation" in protocol:
            raise ValueError("Full-direct checkpoint cannot contain a continuation contract")
    if expected_stage == StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION.value:
        protocol = manifest.get("protocol")
        initialization = protocol.get("initialization") if isinstance(protocol, dict) else None
        continuation = protocol.get("continuation") if isinstance(protocol, dict) else None
        parent_hash = manifest.get("parent_model_sha256")
        if (
            not isinstance(initialization, dict)
            or initialization.get("kind") != "expanded_continuation"
            or not isinstance(continuation, dict)
            or continuation.get("kind") != "scan_expanded_low_rewarm"
            or not isinstance(parent_hash, str)
            or initialization.get("parent_model_sha256") != parent_hash
            or continuation.get("parent_model_sha256") != parent_hash
            or initialization.get("parent_step") != 20_000
            or continuation.get("parent_step") != 20_000
            or continuation.get("sample_step_offset") != 20_000
        ):
            raise ValueError("expanded continuation checkpoint lineage mismatch")
        base_checkpoint = initialization.get("base_checkpoint")
        if not isinstance(base_checkpoint, str) or not Path(base_checkpoint).is_absolute():
            raise ValueError("expanded continuation requires an absolute base checkpoint path")

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
