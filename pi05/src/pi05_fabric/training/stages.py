"""Stage lineage and weight-only transition rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


class StageName(str, Enum):
    INDEPENDENT = "independent"
    RAW_COMMON_STAGE1 = "raw_common_stage1"
    I3_RAW_CORE_STAGE2 = "i3_raw_core_stage2"
    I4_RAW_FULL_STAGE2 = "i4_raw_full_stage2"
    PI_NATIVE_STRONG_INDEPENDENT = "pi_native_strong_independent"
    PI_NATIVE_V2_INDEPENDENT = "pi_native_v2_independent"
    PI_NATIVE_RAW_COMMON_STAGE1 = "pi_native_raw_common_stage1"
    PI_NATIVE_THREE_PATH_STAGE2 = "pi_native_three_path_stage2"
    PI_NATIVE_V2_RAW_COMMON_STAGE1 = "pi_native_v2_raw_common_stage1"
    PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2 = "pi_native_v2_residual_action_stage2"
    PI_NATIVE_V2_FULL_DIRECT = "pi_native_v2_full_direct"
    PI_NATIVE_V2_EXPANDED_CONTINUATION = "pi_native_v2_expanded_continuation"
    PI05_STACKCUBE_3A_COMMON_STAGE1 = "pi05_stackcube_3a_common_stage1"
    PI05_STACKCUBE_3A_FULL_STAGE2 = "pi05_stackcube_3a_full_stage2"
    PI05_FOUR_ARM_COMMON_STAGE1 = "pi05_four_arm_common_stage1"
    PI05_FOUR_ARM_FULL_STAGE2 = "pi05_four_arm_full_stage2"
    PI05_FOUR_ARM_FULL_DIRECT = "pi05_four_arm_full_direct"
    PI05_FRAME4_INDEPENDENT_DIRECT = "pi05_frame4_independent_direct"
    PI05_FRAME4_INDEPENDENT_STAGE1 = "pi05_frame4_independent_stage1"
    PI05_FRAME4_INDEPENDENT_STAGE2 = "pi05_frame4_independent_stage2"
    PI05_FRAME3_COMMON_STAGE1 = "pi05_frame3_common_stage1"
    PI05_FRAME3_FULL_STAGE2 = "pi05_frame3_full_stage2"
    PI05_FRAME3_FULL_DIRECT = "pi05_frame3_full_direct"
    PI05_FRAME3_INDEPENDENT_DIRECT = "pi05_frame3_independent_direct"


@dataclass(frozen=True)
class TrainingSnapshot:
    stage: StageName
    step: int
    schedule_step: int
    params: Any
    opt_state: Any
    rng: jax.Array
    parent_model_sha256: str | None
    model_seed: int | None = None
    training_seed: int | None = None
    protocol_metadata: dict[str, Any] | None = None


def parameter_sha256(params: Any) -> str:
    digest = hashlib.sha256()
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, value in leaves_with_paths:
        array = np.asarray(value)
        digest.update(jax.tree_util.keystr(path).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fork_stage2(
    stage1: TrainingSnapshot,
    *,
    target: StageName,
    optimizer_init: Callable[[Any], Any],
    rng: jax.Array,
    model_seed: int | None = None,
    training_seed: int | None = None,
) -> TrainingSnapshot:
    legacy_fork = (
        stage1.stage is StageName.RAW_COMMON_STAGE1
        and target in (StageName.I3_RAW_CORE_STAGE2, StageName.I4_RAW_FULL_STAGE2)
    )
    native_fork = (
        stage1.stage is StageName.PI_NATIVE_RAW_COMMON_STAGE1
        and target is StageName.PI_NATIVE_THREE_PATH_STAGE2
    )
    native_v2_fork = (
        stage1.stage is StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1
        and target is StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2
    )
    stack_cube_3a_fork = (
        stage1.stage is StageName.PI05_STACKCUBE_3A_COMMON_STAGE1
        and target is StageName.PI05_STACKCUBE_3A_FULL_STAGE2
    )
    four_arm_fork = (
        stage1.stage is StageName.PI05_FOUR_ARM_COMMON_STAGE1
        and target is StageName.PI05_FOUR_ARM_FULL_STAGE2
    )
    frame4_independent_fork = (
        stage1.stage is StageName.PI05_FRAME4_INDEPENDENT_STAGE1
        and target is StageName.PI05_FRAME4_INDEPENDENT_STAGE2
    )
    frame3_full_fork = (
        stage1.stage is StageName.PI05_FRAME3_COMMON_STAGE1
        and target is StageName.PI05_FRAME3_FULL_STAGE2
    )
    native_stage_fork = (
        native_fork
        or native_v2_fork
        or stack_cube_3a_fork
        or four_arm_fork
        or frame4_independent_fork
        or frame3_full_fork
    )
    if not (legacy_fork or native_stage_fork):
        raise ValueError("Stage 2 target is incompatible with its Stage 1 source")
    resolved_model_seed = stage1.model_seed if model_seed is None else model_seed
    if native_stage_fork and training_seed is None:
        raise ValueError("native Stage 2 requires an explicit training seed")
    if native_stage_fork and training_seed == stage1.training_seed:
        raise ValueError("native Stage 2 training seed must differ from Stage 1")
    params = jax.tree.map(lambda value: jnp.array(value), stage1.params)
    return TrainingSnapshot(
        stage=target,
        step=0,
        schedule_step=0,
        params=params,
        opt_state=optimizer_init(params),
        rng=rng,
        parent_model_sha256=parameter_sha256(stage1.params),
        model_seed=resolved_model_seed,
        training_seed=training_seed,
        protocol_metadata=stage1.protocol_metadata,
    )
