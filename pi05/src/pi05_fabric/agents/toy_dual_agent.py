"""Small differentiable model used to gate the dual-agent interaction design."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from pi05_fabric.communication.core import attend_with_remote_private
from pi05_fabric.communication.core import extract_private_kv
from pi05_fabric.communication.core import synchronize_common_hidden


def _independent_copy(tree: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        name: jax.device_put(np.array(value, copy=True))
        for name, value in tree.items()
    }


def initialize_toy_agent_pair(
    key: jax.Array,
    *,
    width: int,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Create equal-valued parameter trees without sharing mutable ownership."""

    if width <= 0:
        raise ValueError("width must be positive")
    keys = jax.random.split(key, 4)
    scale = width**-0.5
    base = {
        name: jax.random.normal(subkey, (width, width)) * scale
        for name, subkey in zip(
            ("encoder", "query", "key", "value"),
            keys,
            strict=True,
        )
    }
    return _independent_copy(base), _independent_copy(base)


def _project_tokens(tokens: jax.Array, matrix: jax.Array) -> jax.Array:
    return jnp.einsum("btd,df->btf", tokens, matrix)


def _project_kv(hidden: jax.Array, matrix: jax.Array) -> jax.Array:
    return _project_tokens(hidden, matrix)[:, None, :, :]


def _project_query(hidden: jax.Array, matrix: jax.Array) -> jax.Array:
    pooled = jnp.mean(hidden, axis=1)
    return jnp.einsum("bd,df->bf", pooled, matrix)[:, None, None, :]


def toy_dual_forward(
    left_params: Mapping[str, jax.Array],
    right_params: Mapping[str, jax.Array],
    left_tokens: jax.Array,
    right_tokens: jax.Array,
    ownership: jax.Array,
    private_indices: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Run one Core-style interaction step for two independent toy agents."""

    left_hidden = _project_tokens(left_tokens, left_params["encoder"])
    right_hidden = _project_tokens(right_tokens, right_params["encoder"])
    left_hidden, right_hidden = synchronize_common_hidden(
        left_hidden,
        right_hidden,
        ownership,
    )

    left_key = _project_kv(left_hidden, left_params["key"])
    left_value = _project_kv(left_hidden, left_params["value"])
    right_key = _project_kv(right_hidden, right_params["key"])
    right_value = _project_kv(right_hidden, right_params["value"])

    left_message = extract_private_kv(left_key, left_value, private_indices)
    right_message = extract_private_kv(right_key, right_value, private_indices)

    left_output = attend_with_remote_private(
        _project_query(left_hidden, left_params["query"]),
        left_key,
        left_value,
        right_message.key,
        right_message.value,
    )
    right_output = attend_with_remote_private(
        _project_query(right_hidden, right_params["query"]),
        right_key,
        right_value,
        left_message.key,
        left_message.value,
    )
    return left_output[:, 0, 0, :], right_output[:, 0, 0, :]
