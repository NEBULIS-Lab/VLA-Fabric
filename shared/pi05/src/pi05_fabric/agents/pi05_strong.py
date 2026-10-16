"""Path-accurate trainable boundary for the native pi0.5 architecture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import nnx


FROZEN_GROUP = "frozen"
ACTION_EXPERT_GROUP = "action_expert_base"
PALIGEMMA_GROUP = "paligemma_kv_norm"
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
) -> str:
    """Assign exactly one optimizer group to a project-owned parameter path."""
    segments = tuple(str(part) for part in path)
    joined = "/".join(segments)

    if any("lora" in segment.lower() for segment in segments):
        return ADAPTER_GROUP
    if any(
        marker in segment
        for segment in segments
        for marker in ("fabric_residual_action", "residual_action")
    ):
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
        return ACTION_EXPERT_GROUP if train_action_ffw else FROZEN_GROUP

    if "/layers/attn/kv_einsum/w" in joined:
        return PALIGEMMA_GROUP if train_paligemma_kv else FROZEN_GROUP
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
) -> Callable[[tuple[Any, ...], Any], bool]:
    def matches(path: tuple[Any, ...], _value: Any) -> bool:
        return (
            native_parameter_group(
                path,
                train_action_ffw=train_action_ffw,
                train_action_attention=train_action_attention,
                train_paligemma_kv=train_paligemma_kv,
            )
            != FROZEN_GROUP
        )

    return matches


def native_trainable_filter(
    *,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
) -> nnx.filterlib.Filter:
    return nnx.All(
        nnx.Param,
        _path_filter(
            train_action_ffw=train_action_ffw,
            train_action_attention=train_action_attention,
            train_paligemma_kv=train_paligemma_kv,
        ),
    )


def native_optimizer_labels(
    state,
    *,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
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
            ),
        )
        for path, _ in state.flat_state().items()
    )
