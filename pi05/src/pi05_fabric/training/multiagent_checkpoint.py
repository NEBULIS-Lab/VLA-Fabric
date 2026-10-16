"""Atomic role-sharded checkpoints for complete multi-agent teams."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from flax import nnx
import jax.numpy as jnp

from pi05_fabric.training.checkpoint import optimizer_state_dict
from pi05_fabric.training.checkpoint import restore_training_state
from pi05_fabric.training.checkpoint import restore_optimizer_state
from pi05_fabric.training.checkpoint import save_training_state
from pi05_fabric.training.multiagent_engine import MultiAgentEngineState
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import TrainingSnapshot
from pi05_fabric.training.stages import parameter_sha256


def _role_path(root: Path, role: int) -> Path:
    if role < 0:
        raise ValueError("role must be non-negative")
    return Path(root) / f"role-{role}"


def save_role_checkpoint(root: Path, *, role: int, snapshot) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    save_training_state(_role_path(root, role), snapshot)


def finalize_team_checkpoint(
    root: Path,
    *,
    agent_count: int,
    protocol_hash: str,
) -> None:
    root = Path(root)
    if (root / "COMPLETE").exists():
        raise FileExistsError(f"team checkpoint is already complete: {root}")
    expected = [_role_path(root, role) for role in range(agent_count)]
    missing = [path.name for path in expected if not (path / "manifest.json").is_file()]
    if missing:
        raise ValueError(f"missing role shards: {', '.join(missing)}")
    role_manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in expected
    ]
    steps = {manifest["step"] for manifest in role_manifests}
    stages = {manifest["stage"] for manifest in role_manifests}
    if len(steps) != 1 or len(stages) != 1:
        raise ValueError("role shards do not share one stage and step")
    for role, manifest in enumerate(role_manifests):
        protocol = manifest.get("protocol") or {}
        if protocol.get("agent_count") != agent_count or protocol.get("role") != role:
            raise ValueError(f"role-{role} protocol metadata is inconsistent")
    team_manifest = {
        "schema_version": 1,
        "agent_count": agent_count,
        "roles": list(range(agent_count)),
        "stage": next(iter(stages)),
        "step": next(iter(steps)),
        "protocol_hash": protocol_hash,
        "role_model_sha256": [manifest["model_sha256"] for manifest in role_manifests],
        "role_state_sha256": [manifest["state_sha256"] for manifest in role_manifests],
    }
    manifest_bytes = (json.dumps(team_manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_tmp = root / "team_manifest.json.tmp"
    with manifest_tmp.open("xb") as stream:
        stream.write(manifest_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    manifest_tmp.replace(root / "team_manifest.json")
    complete_payload = {
        "team_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    complete_tmp = root / "COMPLETE.tmp"
    with complete_tmp.open("x", encoding="utf-8") as stream:
        json.dump(complete_payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    complete_tmp.replace(root / "COMPLETE")


def audit_team_checkpoint_metadata(root: Path, *, expected_agent_count: int) -> dict:
    """Verify team and role metadata without deserializing multi-GiB state."""
    root = Path(root)
    complete_path = root / "COMPLETE"
    manifest_path = root / "team_manifest.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise ValueError("team checkpoint is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_bytes).hexdigest() != complete["team_manifest_sha256"]:
        raise ValueError("team manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["agent_count"] != expected_agent_count:
        raise ValueError("team checkpoint agent count mismatch")
    expected_roles = list(range(expected_agent_count))
    if manifest.get("roles") != expected_roles:
        raise ValueError("team checkpoint roles are inconsistent")
    if len(manifest.get("role_model_sha256", ())) != expected_agent_count:
        raise ValueError("team checkpoint model hashes are incomplete")
    if len(manifest.get("role_state_sha256", ())) != expected_agent_count:
        raise ValueError("team checkpoint state hashes are incomplete")

    for role in manifest["roles"]:
        role_path = _role_path(root, role)
        role_manifest_path = role_path / "manifest.json"
        state_path = role_path / "state.msgpack"
        if not role_manifest_path.is_file() or not state_path.is_file():
            raise ValueError(f"role-{role} checkpoint files are incomplete")
        role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
        protocol = role_manifest.get("protocol") or {}
        if role_manifest["step"] != manifest["step"] or role_manifest["stage"] != manifest["stage"]:
            raise ValueError(f"role-{role} does not match the team manifest")
        if protocol.get("agent_count") != expected_agent_count or protocol.get("role") != role:
            raise ValueError(f"role-{role} protocol metadata is inconsistent")
        if role_manifest["model_sha256"] != manifest["role_model_sha256"][role]:
            raise ValueError(f"role-{role} model hash metadata is inconsistent")
        if role_manifest["state_sha256"] != manifest["role_state_sha256"][role]:
            raise ValueError(f"role-{role} state hash metadata is inconsistent")
    return {**manifest, "complete": True}


def audit_team_checkpoint(root: Path, *, expected_agent_count: int) -> dict:
    manifest = audit_team_checkpoint_metadata(
        root,
        expected_agent_count=expected_agent_count,
    )
    for role in manifest["roles"]:
        snapshot = restore_training_state(_role_path(Path(root), role))
        if snapshot.step != manifest["step"] or snapshot.stage.value != manifest["stage"]:
            raise ValueError(f"role-{role} does not match the team manifest")
    return manifest


def load_role_weights(root: Path, *, role: int, expected_agent_count: int):
    audit_team_checkpoint_metadata(root, expected_agent_count=expected_agent_count)
    if not 0 <= role < expected_agent_count:
        raise ValueError("role is outside the checkpoint team")
    return restore_training_state(_role_path(Path(root), role)).params


def restore_role_snapshot(root: Path, *, role: int, expected_agent_count: int):
    audit_team_checkpoint_metadata(root, expected_agent_count=expected_agent_count)
    if not 0 <= role < expected_agent_count:
        raise ValueError("role is outside the checkpoint team")
    # The serialized-state hash already protects the complete runtime payload.
    # Recomputing the model hash forces another multi-GiB device-to-host pass;
    # the explicit offline team audit retains that deeper provenance check.
    return restore_training_state(
        _role_path(Path(root), role),
        verify_model=False,
    )


def snapshot_role_engine(
    engine,
    *,
    stage: StageName,
    rng,
    parent_model_sha256: str | None,
    model_seed: int,
    training_seed: int,
    protocol_metadata: dict,
) -> TrainingSnapshot:
    return TrainingSnapshot(
        stage=stage,
        step=engine.current_step(),
        schedule_step=engine.current_step(),
        params=engine.selected_params().to_pure_dict(),
        opt_state=optimizer_state_dict(engine.state.opt_state),
        rng=rng,
        parent_model_sha256=parent_model_sha256,
        model_seed=model_seed,
        training_seed=training_seed,
        protocol_metadata=protocol_metadata,
    )


def _replace_selected_params(engine, pure_params) -> None:
    selected = engine.selected_params()
    selected.replace_by_pure_dict(pure_params)
    engine.state = engine.state._replace(model_state=selected)


def resume_role_engine(engine, snapshot: TrainingSnapshot, *, expected_stage: StageName):
    if snapshot.stage is not expected_stage:
        raise ValueError(
            f"expected {expected_stage.value}, found {snapshot.stage.value}"
        )
    _replace_selected_params(engine, snapshot.params)
    engine.state = MultiAgentEngineState(
        engine.state.model_state,
        restore_optimizer_state(engine.state.opt_state, snapshot.opt_state),
        jnp.asarray(snapshot.step, dtype=jnp.int32),
    )
    return snapshot.rng, snapshot.parent_model_sha256


def fork_role_engine_from_stage1(
    engine,
    snapshot: TrainingSnapshot,
    *,
    training_seed: int,
    expected_stage1: StageName = StageName.PI05_STACKCUBE_3A_COMMON_STAGE1,
):
    if snapshot.stage is not expected_stage1:
        raise ValueError(
            f"Stage 2 requires {expected_stage1.value} role shards"
        )
    if snapshot.training_seed == training_seed:
        raise ValueError("Stage 2 training seed must differ from Stage 1")
    parent_hash = parameter_sha256(snapshot.params)
    _replace_selected_params(engine, snapshot.params)
    engine.state = MultiAgentEngineState(
        engine.state.model_state,
        engine.tx.init(engine.state.model_state),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return parent_hash
