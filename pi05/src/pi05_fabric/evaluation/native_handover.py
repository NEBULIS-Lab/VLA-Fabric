"""Evaluation contracts for native pi0.5 Handover checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.training.stages import StageName


@dataclass(frozen=True)
class NativeEvaluationSpec:
    mode: DualPi05Mode
    train_action_ffw: bool = False
    train_action_attention: bool = True
    train_paligemma_kv: bool = True
    train_paligemma_qo: bool = False
    separate_expanded_groups: bool = False
    action_horizon: int = 20
    execution_horizon: int = 20


_SPECS = {
    StageName.PI_NATIVE_V2_INDEPENDENT: NativeEvaluationSpec(
        DualPi05Mode.PI_NATIVE_V2_INDEPENDENT, action_horizon=50, execution_horizon=25
    ),
    StageName.PI_NATIVE_RAW_COMMON_STAGE1: NativeEvaluationSpec(
        DualPi05Mode.PI_NATIVE_RAW_COMMON_ONLY
    ),
    StageName.PI_NATIVE_THREE_PATH_STAGE2: NativeEvaluationSpec(
        DualPi05Mode.PI_NATIVE_THREE_PATH
    ),
    StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1: NativeEvaluationSpec(
        DualPi05Mode.PI_NATIVE_V2_RAW_COMMON_ONLY, action_horizon=50, execution_horizon=25
    ),
    StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2: NativeEvaluationSpec(
        DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION, action_horizon=50, execution_horizon=25
    ),
    StageName.PI_NATIVE_V2_FULL_DIRECT: NativeEvaluationSpec(
        DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION, action_horizon=50, execution_horizon=25
    ),
    StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION: NativeEvaluationSpec(
        DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
        train_action_ffw=True,
        train_paligemma_qo=True,
        separate_expanded_groups=True,
        action_horizon=50,
        execution_horizon=25,
    ),
}


def native_evaluation_spec(stage: StageName) -> NativeEvaluationSpec:
    try:
        return _SPECS[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported native Handover evaluation stage: {stage.value}") from exc


def paired_outcomes(
    stage1: Mapping[int, int], stage2: Mapping[int, int]
) -> dict[str, int]:
    if set(stage1) != set(stage2):
        raise ValueError("paired evaluations must contain identical condition IDs")
    return {
        "both_success": sum(bool(stage1[key]) and bool(stage2[key]) for key in stage1),
        "stage1_only": sum(bool(stage1[key]) and not bool(stage2[key]) for key in stage1),
        "stage2_only": sum(not bool(stage1[key]) and bool(stage2[key]) for key in stage1),
        "both_fail": sum(not bool(stage1[key]) and not bool(stage2[key]) for key in stage1),
    }


def shard_validation_ids(
    condition_ids: tuple[int, ...], *, index: int, count: int
) -> tuple[int, ...]:
    if count <= 0 or index < 0 or index >= count:
        raise ValueError("validation shard requires 0 <= index < count")
    return condition_ids[index::count]


_NATIVE_INFERENCE_MODES = {
    "full": DualPi05Mode.PI_NATIVE_THREE_PATH,
    "core": DualPi05Mode.PI_NATIVE_CORE,
    "common_only": DualPi05Mode.PI_NATIVE_RAW_COMMON_ONLY,
}


_NATIVE_V2_INFERENCE_MODES = {
    "full": DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
    "no_common": DualPi05Mode.PI_NATIVE_V2_WO_COMMON,
    "no_private": DualPi05Mode.PI_NATIVE_V2_WO_PRIVATE,
    "core": DualPi05Mode.PI_NATIVE_V2_CORE,
    "common_only": DualPi05Mode.PI_NATIVE_V2_RAW_COMMON_ONLY,
    "all_off": DualPi05Mode.PI_NATIVE_V2_ALL_OFF,
}

def native_inference_mode(value: str, *, v2: bool = False) -> DualPi05Mode:
    modes = _NATIVE_V2_INFERENCE_MODES if v2 else _NATIVE_INFERENCE_MODES
    try:
        return modes[value]
    except KeyError as exc:
        choices = ", ".join(modes)
        raise ValueError(f"native inference mode must be one of: {choices}") from exc


def diagnostic_validation_ids(parity: str = "even") -> tuple[int, ...]:
    try:
        offset = {"even": 0, "odd": 1}[parity]
    except KeyError as exc:
        raise ValueError("diagnostic parity must be even or odd") from exc
    return tuple(range(offset, 200, 2))


def next_action_index(
    current: int, *, execute_horizon: int, prediction_horizon: int = 20
) -> int:
    if not 1 <= execute_horizon <= prediction_horizon:
        raise ValueError(f"execute_horizon must be between 1 and {prediction_horizon}")
    if not 0 <= current < execute_horizon:
        raise ValueError("current action index is outside execute_horizon")
    return (current + 1) % execute_horizon
