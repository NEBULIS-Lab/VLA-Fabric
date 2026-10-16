"""Functional JAX primitives for the Core dual-agent interface."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pi05_fabric.data.aloha_dual_agent import COMMON
from pi05_fabric.data.aloha_dual_agent import PRIVATE


class PrivateKVMessage(NamedTuple):
    key: jax.Array
    value: jax.Array


def _validate_ownership(ownership: jax.Array, token_count: int) -> None:
    if ownership.ndim != 1 or ownership.shape[0] != token_count:
        raise ValueError(
            f"ownership must have shape ({token_count},), got {ownership.shape}"
        )


def synchronize_common_hidden(
    left_hidden: jax.Array,
    right_hidden: jax.Array,
    ownership: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Average Common positions while preserving each agent's Private states."""

    if left_hidden.shape != right_hidden.shape:
        raise ValueError(
            f"left and right hidden states must have the same shape, got "
            f"{left_hidden.shape} and {right_hidden.shape}"
        )
    if left_hidden.ndim != 3:
        raise ValueError("hidden states must have shape [batch, tokens, width]")
    _validate_ownership(ownership, left_hidden.shape[1])

    common_mask = (ownership == COMMON)[None, :, None]
    averaged = (left_hidden + right_hidden) / 2
    return (
        jnp.where(common_mask, averaged, left_hidden),
        jnp.where(common_mask, averaged, right_hidden),
    )


def private_token_indices(ownership: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    """Resolve the static Private-token layout before entering JAX tracing."""

    ownership_array = np.asarray(ownership)
    if ownership_array.ndim != 1:
        raise ValueError("ownership must be one-dimensional")
    if not np.all(np.isin(ownership_array, (COMMON, PRIVATE))):
        raise ValueError("ownership contains an unknown token class")
    return tuple(int(index) for index in np.flatnonzero(ownership_array == PRIVATE))


def extract_private_kv(
    key: jax.Array,
    value: jax.Array,
    private_indices: tuple[int, ...],
) -> PrivateKVMessage:
    """Create a message containing no Common-token K/V entries."""

    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if key.ndim != 4:
        raise ValueError("key and value must have shape [batch, heads, tokens, width]")
    if any(index < 0 or index >= key.shape[-2] for index in private_indices):
        raise ValueError("private token index is outside the K/V sequence")
    index_array = jnp.asarray(private_indices, dtype=jnp.int32)
    return PrivateKVMessage(
        key=jnp.take(key, index_array, axis=-2),
        value=jnp.take(value, index_array, axis=-2),
    )


def attend_with_remote_private(
    query: jax.Array,
    local_key: jax.Array,
    local_value: jax.Array,
    remote_private_key: jax.Array,
    remote_private_value: jax.Array,
) -> jax.Array:
    """Attend with a receiver-local query over local and remote Private K/V."""

    if local_key.shape != local_value.shape:
        raise ValueError("local key and value must have the same shape")
    if remote_private_key.shape != remote_private_value.shape:
        raise ValueError("remote key and value must have the same shape")
    if query.ndim != 4 or local_key.ndim != 4 or remote_private_key.ndim != 4:
        raise ValueError("query, key, and value tensors must be rank four")
    if query.shape[:2] != local_key.shape[:2] or query.shape[:2] != remote_private_key.shape[:2]:
        raise ValueError("batch and head dimensions must match")
    if query.shape[-1] != local_key.shape[-1] or query.shape[-1] != remote_private_key.shape[-1]:
        raise ValueError("attention widths must match")

    key = jnp.concatenate([local_key, remote_private_key], axis=-2)
    value = jnp.concatenate([local_value, remote_private_value], axis=-2)
    scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(query.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, value)
