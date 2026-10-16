from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0_config

from pi05_fabric.agents.dual_pi05 import DualPi05, DualPi05Mode
from pi05_fabric.communication.paired_gemma import action_suffix_kv


def test_action_suffix_kv_uses_only_final_action_horizon_tokens():
    key = jnp.arange(2 * 7 * 1 * 3, dtype=jnp.float32).reshape(2, 7, 1, 3)
    value = -key

    selected_key, selected_value = action_suffix_kv(key, value, action_horizon=4)

    assert selected_key.shape == (2, 4, 1, 3)
    assert selected_value.shape == (2, 4, 1, 3)
    assert jnp.array_equal(selected_key, key[:, -4:])
    assert jnp.array_equal(selected_value, value[:, -4:])


def test_action_suffix_kv_rejects_invalid_horizon():
    key = jnp.zeros((1, 3, 1, 2))
    value = jnp.zeros_like(key)
    for horizon in (0, 4):
        try:
            action_suffix_kv(key, value, action_horizon=horizon)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid action_horizon={horizon} to fail")


def test_native_mode_uses_remote_action_kv_without_final_linear_module():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    pair = DualPi05(
        config.create(jax.random.key(1)),
        config.create(jax.random.key(2)),
        mode=DualPi05Mode.PI_NATIVE_THREE_PATH,
        rngs=nnx.Rngs(3),
    )

    assert tuple(DualPi05Mode.PI_NATIVE_THREE_PATH.interaction_spec) == (
        True,
        True,
        True,
        False,
    )
    assert not hasattr(pair, "fabric_linear_action_left")
    assert not hasattr(pair, "fabric_linear_action_right")


def test_all_native_modes_omit_the_legacy_linear_module():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    for index, mode in enumerate(
        (
            DualPi05Mode.PI_NATIVE_INDEPENDENT,
            DualPi05Mode.PI_NATIVE_RAW_COMMON_ONLY,
            DualPi05Mode.PI_NATIVE_CORE,
            DualPi05Mode.PI_NATIVE_THREE_PATH,
        )
    ):
        pair = DualPi05(
            config.create(jax.random.key(10 + 2 * index)),
            config.create(jax.random.key(11 + 2 * index)),
            mode=mode,
            rngs=nnx.Rngs(20 + index),
        )
        assert not hasattr(pair, "fabric_linear_action_left")
        assert not hasattr(pair, "fabric_linear_action_right")


def test_native_core_keeps_common_and_private_but_disables_remote_action():
    assert tuple(DualPi05Mode.PI_NATIVE_CORE.interaction_spec) == (
        True,
        True,
        False,
        False,
    )
    assert DualPi05Mode.PI_NATIVE_CORE.is_native
