"""Path-accurate trainable boundary for the native pi0.5 architecture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import nnx
import numpy as np


FROZEN_GROUP = "frozen"
ACTION_EXPERT_GROUP = "action_expert_base"
ACTION_EXPERT_FFW_GROUP = "action_expert_ffw_base"
PALIGEMMA_GROUP = "paligemma_kv_norm"
PALIGEMMA_QO_GROUP = "paligemma_qo_base"
ADAPTER_GROUP = "lora_and_projections"

_ACTION_PROJECTIONS = frozenset(
    {"action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj"}
)


def native_parameter_group(
    path: tuple[Any, ...],
    *,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
    train_paligemma_qo: bool = False,
    separate_expanded_groups: bool = False,
) -> str:
    """Assign exactly one optimizer group to a project-owned parameter path."""
    segments = tuple(str(part) for part in path)
    joined = "/".join(segments)

    if any("lora" in segment.lower() for segment in segments):
        return ADAPTER_GROUP
    if any("residual_action" in segment for segment in segments):
        return ADAPTER_GROUP
    if any(segment in _ACTION_PROJECTIONS for segment in segments):
        return ADAPTER_GROUP
    if "PaliGemma/llm/" not in joined:
        return FROZEN_GROUP

    if "/layers/attn/" in joined and any(
        marker in joined
        for marker in (
            "q_einsum_1/w",
            "kv_einsum_1/w",
            "attn_vec_einsum_1/w",
        )
    ):
        return ACTION_EXPERT_GROUP if train_action_attention else FROZEN_GROUP
    if any(
        marker in joined
        for marker in (
            "pre_attention_norm_1/",
            "pre_ffw_norm_1/",
            "final_norm_1/",
        )
    ):
        return ACTION_EXPERT_GROUP
    if "/layers/mlp_1/" in joined:
        if not train_action_ffw:
            return FROZEN_GROUP
        return ACTION_EXPERT_FFW_GROUP if separate_expanded_groups else ACTION_EXPERT_GROUP

    if "/layers/attn/kv_einsum/w" in joined:
        return PALIGEMMA_GROUP if train_paligemma_kv else FROZEN_GROUP
    if "/layers/attn/" in joined and any(
        marker in joined
        for marker in (
            "/layers/attn/q_einsum/w",
            "/layers/attn/attn_vec_einsum/w",
        )
    ):
        return PALIGEMMA_QO_GROUP if train_paligemma_qo else FROZEN_GROUP
    if any(
        marker in joined
        for marker in (
            "/layers/pre_attention_norm/scale",
            "/layers/pre_ffw_norm/scale",
            "/final_norm/scale",
        )
    ):
        return PALIGEMMA_GROUP
    return FROZEN_GROUP


def _path_filter(
    *,
    train_action_ffw: bool,
    train_action_attention: bool,
    train_paligemma_kv: bool,
    train_paligemma_qo: bool,
    separate_expanded_groups: bool,
) -> Callable[[tuple[Any, ...], Any], bool]:
    def matches(path: tuple[Any, ...], _value: Any) -> bool:
        return (
            native_parameter_group(
                path,
                train_action_ffw=train_action_ffw,
                train_action_attention=train_action_attention,
                train_paligemma_kv=train_paligemma_kv,
                train_paligemma_qo=train_paligemma_qo,
                separate_expanded_groups=separate_expanded_groups,
            )
            != FROZEN_GROUP
        )

    return matches


def native_trainable_filter(
    *,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
    train_paligemma_qo: bool = False,
    separate_expanded_groups: bool = False,
) -> nnx.filterlib.Filter:
    return nnx.All(
        nnx.Param,
        _path_filter(
            train_action_ffw=train_action_ffw,
            train_action_attention=train_action_attention,
            train_paligemma_kv=train_paligemma_kv,
            train_paligemma_qo=train_paligemma_qo,
            separate_expanded_groups=separate_expanded_groups,
        ),
    )


def native_optimizer_labels(
    state,
    *,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
    train_paligemma_qo: bool = False,
    separate_expanded_groups: bool = False,
):
    """Build an Optax-compatible label tree for an already filtered NNX state."""
    return nnx.State.from_flat_path(
        (
            path,
            native_parameter_group(
                path,
                train_action_ffw=train_action_ffw,
                train_action_attention=train_action_attention,
                train_paligemma_kv=train_paligemma_kv,
                train_paligemma_qo=train_paligemma_qo,
                separate_expanded_groups=separate_expanded_groups,
            ),
        )
        for path, _ in state.flat_state().items()
    )


def native_group_summary(
    state,
    *,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
    train_paligemma_qo: bool = False,
    separate_expanded_groups: bool = False,
) -> dict[str, dict[str, int]]:
    """Summarize the selected parameter boundary by optimizer group."""
    summary: dict[str, dict[str, int]] = {}
    for path, variable in state.flat_state().items():
        group = native_parameter_group(
            path,
            train_action_ffw=train_action_ffw,
            train_action_attention=train_action_attention,
            train_paligemma_kv=train_paligemma_kv,
            train_paligemma_qo=train_paligemma_qo,
            separate_expanded_groups=separate_expanded_groups,
        )
        if group == FROZEN_GROUP:
            raise ValueError(f"selected state contains frozen parameter: {'/'.join(map(str, path))}")
        value = getattr(variable, "value", variable)
        count = int(np.prod(value.shape, dtype=np.int64))
        record = summary.setdefault(group, {"tensor_count": 0, "parameter_count": 0, "bytes": 0})
        record["tensor_count"] += 1
        record["parameter_count"] += count
        record["bytes"] += count * np.dtype(value.dtype).itemsize
    return summary
