"""Checkpoint semantics for same-stage resume and Stage1-to-Stage2 forks."""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp

from pi05_fabric.training.checkpoint import optimizer_state_dict
from pi05_fabric.training.checkpoint import save_training_state
from pi05_fabric.training.checkpoint import restore_optimizer_state
from pi05_fabric.training.engine import EngineState
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import TrainingSnapshot
from pi05_fabric.training.stages import parameter_sha256


def snapshot_engine(
    engine,
    *,
    stage: StageName,
    rng: jax.Array,
    parent_model_sha256: str | None = None,
    model_seed: int | None = None,
    training_seed: int | None = None,
    protocol_metadata: dict | None = None,
) -> TrainingSnapshot:
    state = engine.host_state()
    return TrainingSnapshot(
        stage=stage,
        step=int(state.step),
        schedule_step=int(state.step),
        params=engine.selected_params(state).to_pure_dict(),
        opt_state=optimizer_state_dict(state.opt_state),
        rng=rng,
        parent_model_sha256=parent_model_sha256,
        model_seed=model_seed,
        training_seed=training_seed,
        protocol_metadata=protocol_metadata,
    )


def write_engine_checkpoint(
    path,
    engine,
    *,
    stage: StageName,
    rng: jax.Array,
    parent_model_sha256: str | None = None,
    model_seed: int | None = None,
    training_seed: int | None = None,
    protocol_metadata: dict | None = None,
) -> None:
    """Serialize one engine snapshot without retaining its device-backed tree."""
    snapshot = snapshot_engine(
        engine,
        stage=stage,
        rng=rng,
        parent_model_sha256=parent_model_sha256,
        model_seed=model_seed,
        training_seed=training_seed,
        protocol_metadata=protocol_metadata,
    )
    try:
        save_training_state(path, snapshot)
    finally:
        del snapshot
        gc.collect()


def _replace_selected_params(engine, pure_params) -> None:
    selected = engine.selected_params()
    selected.replace_by_pure_dict(pure_params)
    engine.state = engine.state._replace(model_state=selected)


def apply_same_stage_snapshot(engine, snapshot: TrainingSnapshot, *, expected_stage: StageName) -> None:
    if snapshot.stage is not expected_stage:
        raise ValueError(f"expected {expected_stage.value}, found {snapshot.stage.value}")
    _replace_selected_params(engine, snapshot.params)
    engine.state = EngineState(
        engine.state.model_state,
        restore_optimizer_state(engine.state.opt_state, snapshot.opt_state),
        jnp.asarray(snapshot.step, dtype=jnp.int32),
    )


def fork_from_stage1(
    engine,
    snapshot: TrainingSnapshot,
    *,
    target: StageName,
    rng: jax.Array,
    model_seed: int | None = None,
    training_seed: int | None = None,
    protocol_metadata: dict | None = None,
) -> TrainingSnapshot:
    legacy_fork = (
        snapshot.stage is StageName.RAW_COMMON_STAGE1
        and target in (StageName.I3_RAW_CORE_STAGE2, StageName.I4_RAW_FULL_STAGE2)
    )
    native_fork = (
        snapshot.stage is StageName.PI_NATIVE_RAW_COMMON_STAGE1
        and target is StageName.PI_NATIVE_THREE_PATH_STAGE2
    )
    native_v2_fork = (
        snapshot.stage is StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1
        and target is StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2
    )
    if not (legacy_fork or native_fork or native_v2_fork):
        raise ValueError("Stage 2 target is incompatible with its Stage 1 source")
    if (native_fork or native_v2_fork) and training_seed is None:
        raise ValueError("native Stage 2 requires an explicit training seed")
    if (native_fork or native_v2_fork) and training_seed == snapshot.training_seed:
        raise ValueError("native Stage 2 training seed must differ from Stage 1")
    parent_hash = parameter_sha256(snapshot.params)
    _replace_selected_params(engine, snapshot.params)
    selected = engine.selected_params()
    engine.state = EngineState(
        engine.state.model_state,
        engine.tx.init(selected),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return snapshot_engine(
        engine,
        stage=target,
        rng=rng,
        parent_model_sha256=parent_hash,
        model_seed=snapshot.model_seed if model_seed is None else model_seed,
        training_seed=training_seed,
        protocol_metadata=(snapshot.protocol_metadata if protocol_metadata is None else protocol_metadata),
    )
