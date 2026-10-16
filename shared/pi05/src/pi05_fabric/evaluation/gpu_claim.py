"""Early JAX device reservation for race-free long-running evaluation launch."""

from __future__ import annotations

from typing import Any

import numpy as np


def claim_jax_device(*, jax_module: Any | None = None):
    """Trigger the configured JAX allocator before checkpoint loading."""
    if jax_module is None:
        import jax as jax_module

    devices = jax_module.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(f"early GPU claim requires exactly one visible GPU, got {len(devices)}")
    token = jax_module.device_put(np.zeros((1,), dtype=np.float32), devices[0])
    token.block_until_ready()
    return token
