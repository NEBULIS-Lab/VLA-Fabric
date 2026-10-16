"""Complete same-stage checkpoint save and restore."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from flax import nnx
from flax import serialization
import jax
import jax.numpy as jnp

from pi05_fabric.training.stages import StageName, TrainingSnapshot, parameter_sha256


def _strip_nnx_states(value: Any) -> Any:
    if isinstance(value, nnx.State):
        return jax.tree.map(jnp.asarray, value.to_pure_dict())
    if isinstance(value, dict):
        return {key: _strip_nnx_states(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_nnx_states(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_nnx_states(item) for item in value)
    return value


def optimizer_state_dict(opt_state: Any) -> dict[str, Any]:
    """Return a msgpack-safe optimizer state with NNX wrappers removed."""
    return _strip_nnx_states(serialization.to_state_dict(opt_state))


def _hydrate_nnx_states(template: Any, saved: Any) -> Any:
    if isinstance(template, nnx.State):
        restored = nnx.State(template)
        restored.replace_by_pure_dict(saved)
        return restored
    if isinstance(template, dict):
        return {key: _hydrate_nnx_states(template[key], saved[key]) for key in template}
    if isinstance(template, list):
        return [_hydrate_nnx_states(item, saved[index]) for index, item in enumerate(template)]
    return saved


def restore_optimizer_state(template: Any, state_dict: dict[str, Any]) -> Any:
    """Rebuild Optax and NNX container types from a pure checkpoint tree."""
    template_dict = serialization.to_state_dict(template)
    hydrated = _hydrate_nnx_states(template_dict, state_dict)
    return serialization.from_state_dict(template, hydrated)


def save_training_state(path: Path, snapshot: TrainingSnapshot) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        payload = {
            "params": snapshot.params,
            "opt_state": optimizer_state_dict(snapshot.opt_state),
            "rng_data": jax.random.key_data(snapshot.rng),
        }
        state_bytes = serialization.msgpack_serialize(payload)
        with (temporary / "state.msgpack").open("xb") as state_file:
            state_file.write(state_bytes)
            state_file.flush()
            os.fsync(state_file.fileno())
        manifest = {
            "schema_version": 3,
            "stage": snapshot.stage.value,
            "step": snapshot.step,
            "schedule_step": snapshot.schedule_step,
            "parent_model_sha256": snapshot.parent_model_sha256,
            "model_sha256": parameter_sha256(snapshot.params),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "model_seed": snapshot.model_seed,
            "training_seed": snapshot.training_seed,
            "protocol": snapshot.protocol_metadata,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        with (temporary / "manifest.json").open("xb") as manifest_file:
            manifest_file.write(manifest_bytes)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.rename(path)
        temporary = None
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def restore_training_state(path: Path) -> TrainingSnapshot:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    state_bytes = (path / "state.msgpack").read_bytes()
    if hashlib.sha256(state_bytes).hexdigest() != manifest["state_sha256"]:
        raise ValueError("checkpoint state hash mismatch")
    payload = serialization.msgpack_restore(state_bytes)
    params = jax.tree.map(jnp.asarray, payload["params"])
    if parameter_sha256(params) != manifest["model_sha256"]:
        raise ValueError("checkpoint model hash mismatch")
    return TrainingSnapshot(
        stage=StageName(manifest["stage"]),
        step=int(manifest["step"]),
        schedule_step=int(manifest["schedule_step"]),
        params=params,
        opt_state=jax.tree.map(jnp.asarray, payload["opt_state"]),
        rng=jax.random.wrap_key_data(jnp.asarray(payload["rng_data"], dtype=jnp.uint32)),
        parent_model_sha256=manifest["parent_model_sha256"],
        model_seed=manifest.get("model_seed"),
        training_seed=manifest.get("training_seed"),
        protocol_metadata=manifest.get("protocol"),
    )
