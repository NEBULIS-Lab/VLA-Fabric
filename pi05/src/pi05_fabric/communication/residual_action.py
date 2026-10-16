"""Receiver-local residual attention parameters for PI0.5 Native V2."""

from __future__ import annotations

import math

from flax import nnx
import jax
import jax.numpy as jnp


class ResidualActionBranch(nnx.Module):
    """Own one independent projection and zero-initialized gate per layer."""

    def __init__(
        self,
        *,
        depth: int,
        width: int,
        input_width: int | None = None,
        rngs: nnx.Rngs,
    ):
        input_width = width if input_width is None else input_width
        if depth <= 0 or width <= 0 or input_width <= 0:
            raise ValueError("residual action dimensions must be positive")
        kernel = jax.random.normal(
            rngs.params(), (depth, input_width, width), dtype=jnp.float32
        ) / math.sqrt(input_width)
        self.kernel = nnx.Param(kernel)
        self.gate = nnx.Param(jnp.zeros((depth,), dtype=jnp.float32))
