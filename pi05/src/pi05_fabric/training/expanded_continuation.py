"""Weight-only continuation into an explicitly expanded trainable boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import parameter_sha256


def _flat_by_path(tree):
    return {
        jax.tree_util.keystr(path): (path, value)
        for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def merge_parent_params_into_expanded(current, parent):
    """Overlay every parent trainable on a larger base-initialized boundary."""
    current_flat = _flat_by_path(current)
    parent_flat = _flat_by_path(parent)
    for key, (_path, parent_value) in parent_flat.items():
        if key not in current_flat:
            raise ValueError(f"parent trainable {key} is missing from expanded boundary")
        _, current_value = current_flat[key]
        if np.shape(parent_value) != np.shape(current_value) or parent_value.dtype != current_value.dtype:
            raise ValueError(f"parent trainable {key} has changed shape or dtype")

    parent_values = {key: value for key, (_path, value) in parent_flat.items()}
    return jax.tree_util.tree_map_with_path(
        lambda path, value: parent_values.get(jax.tree_util.keystr(path), value),
        current,
    )


def prepare_expanded_continuation(path, args, config, protocol):
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    if spec.get("schema_version") != 1 or spec.get("kind") != "scan_expanded_low_rewarm":
        raise ValueError("unsupported expanded continuation contract")
    if config.stage is not StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION:
        raise ValueError("expanded continuation requires its dedicated training stage")
    if config.stage1_checkpoint is not None:
        raise ValueError("expanded continuation cannot use a Stage 1 parent")

    expected_parent = Path(spec["parent_checkpoint"]).resolve()
    if config.weights_checkpoint is not None and config.weights_checkpoint.resolve() != expected_parent:
        raise ValueError("wrong expanded continuation parent path")

    for name, expected in spec["trainer_contract"].items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(f"expanded continuation requires {name}={expected}, found {actual}")

    parent = json.loads((expected_parent / "manifest.json").read_text())
    expected_identity = {
        "stage": spec["parent_stage"],
        "step": spec["parent_step"],
        "schedule_step": spec["parent_step"],
        "model_sha256": spec["parent_model_sha256"],
        "model_seed": config.resolved_model_seed,
        "training_seed": config.resolved_training_seed,
    }
    if any(parent.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("expanded continuation parent manifest mismatch")
    parent_protocol = parent.get("protocol", {})
    if parent_protocol.get("initialization", {}).get("base_tree_sha256") != spec["base_tree_sha256"]:
        raise ValueError("expanded continuation parent manifest has the wrong base")
    for key in (
        "action_horizon",
        "execution_horizon",
        "flow_steps",
        "normalization_file",
        "normalization",
    ):
        if parent_protocol.get(key) != protocol.get(key):
            raise ValueError(f"expanded continuation parent protocol changed: {key}")
    if spec["sample_step_offset"] != spec["parent_step"]:
        raise ValueError("expanded continuation must preserve the parent sample stream")
    if spec["cumulative_target_step"] != spec["parent_step"] + config.steps:
        raise ValueError("expanded continuation cumulative target is inconsistent")

    return {
        "kind": spec["kind"],
        "spec_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_checkpoint": str(expected_parent),
        "parent_stage": spec["parent_stage"],
        "parent_step": spec["parent_step"],
        "parent_model_sha256": spec["parent_model_sha256"],
        "base_tree_sha256": spec["base_tree_sha256"],
        "sample_step_offset": spec["sample_step_offset"],
        "cumulative_target_step": spec["cumulative_target_step"],
        "fresh_optimizer": True,
        "restore_parent_rng": True,
        "trainer_contract": spec["trainer_contract"],
    }


def apply_expanded_continuation_weights(engine, snapshot, *, contract, protocol, model_seed):
    if engine.current_step() != 0:
        raise ValueError("expanded continuation requires a fresh optimizer")
    if snapshot.stage is not StageName.PI_NATIVE_V2_FULL_DIRECT:
        raise ValueError("expanded continuation requires a Full-direct parent")
    if snapshot.step != contract["parent_step"] or snapshot.schedule_step != snapshot.step:
        raise ValueError("expanded continuation parent step mismatch")
    if snapshot.model_seed != model_seed or snapshot.training_seed != protocol["training_contract"]["training_seed"]:
        raise ValueError("expanded continuation parent seed mismatch")
    if parameter_sha256(snapshot.params) != contract["parent_model_sha256"]:
        raise ValueError("expanded continuation parent parameter hash mismatch")
    parent_protocol = snapshot.protocol_metadata or {}
    for key in (
        "action_horizon",
        "execution_horizon",
        "flow_steps",
        "normalization_file",
        "normalization",
    ):
        if parent_protocol.get(key) != protocol.get(key):
            raise ValueError(f"expanded continuation parent protocol changed: {key}")

    from pi05_fabric.training.engine_checkpoint import _replace_selected_params

    current = engine.selected_params().to_pure_dict()
    merged = merge_parent_params_into_expanded(current, snapshot.params)
    _replace_selected_params(engine, merged)
    return snapshot.rng


def validate_expanded_resume(snapshot, *, protocol, training_seed, model_seed):
    if snapshot.stage is not StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION:
        raise ValueError("expanded resume requires a same-stage checkpoint")
    if snapshot.protocol_metadata != protocol:
        raise ValueError("expanded resume contract changed")
    if snapshot.training_seed != training_seed or snapshot.model_seed != model_seed:
        raise ValueError("expanded resume seed changed")
    if snapshot.parent_model_sha256 != protocol["continuation"]["parent_model_sha256"]:
        raise ValueError("expanded resume parent lineage changed")
    if snapshot.schedule_step != snapshot.step:
        raise ValueError("expanded resume local schedule step changed")
