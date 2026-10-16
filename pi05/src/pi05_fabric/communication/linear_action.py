"""Receiver-local Linear Action interaction used by I4-RAW Full."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ReceiverLinearParams(NamedTuple):
    kernel: jax.Array
    bias: jax.Array


def init_receiver_linear(
    width: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> ReceiverLinearParams:
    """Initialize [local; remote] projection as an exact local identity."""

    if width <= 0:
        raise ValueError("width must be positive")
    local = jnp.eye(width, dtype=dtype)
    remote = jnp.zeros((width, width), dtype=dtype)
    return ReceiverLinearParams(
        kernel=jnp.concatenate([local, remote], axis=0),
        bias=jnp.zeros((width,), dtype=dtype),
    )


def fuse_receiver_linear(
    local_hidden: jax.Array,
    remote_hidden: jax.Array,
    params: ReceiverLinearParams,
) -> jax.Array:
    """Fuse aligned action-expert hidden states on the receiver."""

    if local_hidden.shape != remote_hidden.shape:
        raise ValueError("local and remote hidden states must have the same shape")
    if local_hidden.ndim != 3:
        raise ValueError(
            "action hidden states must have shape [batch, horizon, width]"
        )
    width = local_hidden.shape[-1]
    if params.kernel.shape != (2 * width, width):
        raise ValueError(
            f"kernel must have shape ({2 * width}, {width}), got {params.kernel.shape}"
        )
    if params.bias.shape != (width,):
        raise ValueError(f"bias must have shape ({width},), got {params.bias.shape}")

    paired = jnp.concatenate([local_hidden, remote_hidden], axis=-1)
    return jnp.einsum("btd,df->btf", paired, params.kernel) + params.bias


def fuse_action_pair(
    left_hidden: jax.Array,
    right_hidden: jax.Array,
    left_params: ReceiverLinearParams,
    right_params: ReceiverLinearParams,
    *,
    enabled: bool,
) -> tuple[jax.Array, jax.Array]:
    """Apply separate receiver-local projections without a joint action head."""

    if left_hidden.shape != right_hidden.shape:
        raise ValueError("left and right hidden states must have the same shape")
    if not enabled:
        return left_hidden, right_hidden
    return (
        fuse_receiver_linear(left_hidden, right_hidden, left_params),
        fuse_receiver_linear(right_hidden, left_hidden, right_params),
    )
