"""Operator-level symmetric Common fusion for the selected RAW interface."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pi05_fabric.data.aloha_dual_agent import COMMON


RAW_FUSION_POINTS = (
    "pre_attention_norm",
    "query",
    "key",
    "value",
    "attention_output",
    "pre_ffw_norm",
    "ffw_output",
)


def fuse_raw_common_operators(
    left_operators: jax.Array,
    right_operators: jax.Array,
    ownership: jax.Array,
    *,
    enabled: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fuse seven aligned operator tensors at Common token positions.

    The leading axis follows RAW_FUSION_POINTS and the token axis is 2.
    All remaining axes are treated as operator feature dimensions.
    """

    if left_operators.shape != right_operators.shape:
        raise ValueError(
            "left and right RAW operator tensors must have identical shapes"
        )
    if left_operators.ndim < 4:
        raise ValueError(
            "RAW operator tensors must have shape [events, batch, tokens, ...]"
        )
    if left_operators.shape[0] != len(RAW_FUSION_POINTS):
        raise ValueError("expected seven RAW operator tensors")
    if ownership.ndim == 1:
        valid = ownership.shape[0] == left_operators.shape[2]
        mask_shape = (1, 1, ownership.shape[0]) + (1,) * (left_operators.ndim - 3)
    elif ownership.ndim == 2:
        valid = ownership.shape == left_operators.shape[1:3]
        mask_shape = (1, *ownership.shape) + (1,) * (left_operators.ndim - 3)
    else:
        valid, mask_shape = False, ()
    if not valid:
        raise ValueError(
            "ownership must have shape [tokens] or [batch, tokens] matching the operator tensor; "
            f"got {ownership.shape}"
        )

    if not enabled:
        counts = jnp.zeros((len(RAW_FUSION_POINTS),), dtype=jnp.int32)
        return left_operators, right_operators, counts

    common_mask = (ownership == COMMON).reshape(mask_shape)
    averaged = (left_operators + right_operators) / 2
    left = jnp.where(common_mask, averaged, left_operators)
    right = jnp.where(common_mask, averaged, right_operators)
    counts = jnp.ones((len(RAW_FUSION_POINTS),), dtype=jnp.int32)
    return left, right, counts

