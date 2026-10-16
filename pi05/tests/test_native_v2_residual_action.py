from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models import pi0_config

from pi05_fabric.agents.dual_pi05 import DualPi05
from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.agents.pi05_strong import ADAPTER_GROUP
from pi05_fabric.agents.pi05_strong import native_parameter_group
from pi05_fabric.communication.paired_gemma import paired_gemma_forward


def _residual_action_branch():
    from pi05_fabric.communication.residual_action import ResidualActionBranch

    return ResidualActionBranch


def _toy():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    left = config.create(jax.random.key(10))
    right = config.create(jax.random.key(11))
    configs = (gemma.get_config("dummy"), gemma.get_config("dummy"))
    return config, left, right, configs


def _inputs(offset):
    prefix = jax.random.normal(jax.random.key(20 + offset), (1, 4, 64))
    suffix = jax.random.normal(jax.random.key(30 + offset), (1, 2, 64))
    positions = jnp.arange(6, dtype=jnp.int32)[None, :]
    mask = jnp.ones((1, 6, 6), dtype=bool).at[:, :4, 4:].set(False)
    cond = jax.random.normal(jax.random.key(40 + offset), (1, 64))
    return (prefix, suffix), positions, mask, (None, cond)


def _forward_kwargs():
    left_inputs, left_positions, left_mask, left_cond = _inputs(0)
    right_inputs, right_positions, right_mask, right_cond = _inputs(1)
    return left_inputs, right_inputs, {
        "left_positions": left_positions,
        "right_positions": right_positions,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "left_adarms_cond": left_cond,
        "right_adarms_cond": right_cond,
        "ownership": jnp.array([0, 0, 1, 1], dtype=jnp.int8),
        "common_enabled": True,
        "private_kv_enabled": True,
        "action_horizon": 2,
    }


def test_v2_mode_uses_residual_action_without_changing_v1_interaction_tuple():
    config, left, right, _ = _toy()
    pair = DualPi05(
        left,
        right,
        mode=DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
        rngs=nnx.Rngs(3),
    )

    assert tuple(DualPi05Mode.PI_NATIVE_THREE_PATH.interaction_spec) == (
        True,
        True,
        True,
        False,
    )
    assert tuple(DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION.interaction_spec) == (
        True,
        True,
        False,
        False,
    )
    assert DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION.remote_action_residual
    assert hasattr(pair, "fabric_residual_action_left")
    assert hasattr(pair, "fabric_residual_action_right")
    assert not hasattr(pair, "fabric_linear_action_left")
    assert config.action_horizon == 2


def test_v2_independent_has_no_communication_modules():
    config, left, right, _ = _toy()
    pair = DualPi05(
        left,
        right,
        mode=DualPi05Mode.PI_NATIVE_V2_INDEPENDENT,
        rngs=nnx.Rngs(30),
    )

    mode = DualPi05Mode.PI_NATIVE_V2_INDEPENDENT
    assert mode.is_v2
    assert tuple(mode.interaction_spec) == (False, False, False, False)
    assert not mode.remote_action_residual
    assert not hasattr(pair, "fabric_residual_action_left")
    assert not hasattr(pair, "fabric_residual_action_right")
    assert not hasattr(pair, "fabric_linear_action_left")
    assert not hasattr(pair, "fabric_linear_action_right")
    assert config.action_horizon == 2


def test_receiver_local_residual_parameters_are_independent_and_zero_gated():
    left = _residual_action_branch()(depth=4, width=64, rngs=nnx.Rngs(1))
    right = _residual_action_branch()(depth=4, width=64, rngs=nnx.Rngs(2))
    left_state = nnx.state(left).to_pure_dict()
    right_state = nnx.state(right).to_pure_dict()

    np.testing.assert_array_equal(left_state["gate"], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(right_state["gate"], np.zeros(4, dtype=np.float32))
    assert float(jnp.linalg.norm(left_state["kernel"])) > 0
    assert not np.array_equal(left_state["kernel"], right_state["kernel"])


def test_residual_parameters_belong_to_adapter_optimizer_group():
    assert (
        native_parameter_group(
            ("fabric_residual_action_left", "kernel"),
            train_action_ffw=False,
        )
        == ADAPTER_GROUP
    )
    assert (
        native_parameter_group(
            ("fabric_residual_action_right", "gate"),
            train_action_ffw=False,
        )
        == ADAPTER_GROUP
    )


def test_zero_residual_gate_matches_core_and_keeps_action_kv_out_of_core_softmax():
    _, left, right, configs = _toy()
    left_inputs, right_inputs, kwargs = _forward_kwargs()
    left_params = nnx.state(left).to_pure_dict()["PaliGemma"]["llm"]
    right_params = nnx.state(right).to_pure_dict()["PaliGemma"]["llm"]
    left_branch = nnx.state(
        _residual_action_branch()(depth=4, width=64, input_width=128, rngs=nnx.Rngs(4))
    ).to_pure_dict()
    right_branch = nnx.state(
        _residual_action_branch()(depth=4, width=64, input_width=128, rngs=nnx.Rngs(5))
    ).to_pure_dict()

    core_left, core_right, _ = paired_gemma_forward(
        left_params,
        right_params,
        configs,
        left_inputs,
        right_inputs,
        remote_action_kv_enabled=False,
        **kwargs,
    )
    v2_left, v2_right, trace = paired_gemma_forward(
        left_params,
        right_params,
        configs,
        left_inputs,
        right_inputs,
        remote_action_kv_enabled=False,
        remote_action_residual_enabled=True,
        left_residual_params=left_branch,
        right_residual_params=right_branch,
        **kwargs,
    )

    for actual, expected in zip(v2_left, core_left, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=0,
            atol=2e-3,
        )
    for actual, expected in zip(v2_right, core_right, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=0,
            atol=2e-3,
        )
    assert trace.remote_action_tokens == 0
    assert trace.remote_action_residual_tokens == 2
    assert trace.transmitted_queries == 0
    assert trace.transmitted_raw_observations == 0


def test_zero_gate_has_gradient_and_nonzero_gate_changes_action_suffix():
    _, left, right, configs = _toy()
    left_inputs, right_inputs, kwargs = _forward_kwargs()
    left_params = nnx.state(left).to_pure_dict()["PaliGemma"]["llm"]
    right_params = nnx.state(right).to_pure_dict()["PaliGemma"]["llm"]
    left_branch = nnx.state(
        _residual_action_branch()(depth=4, width=64, input_width=128, rngs=nnx.Rngs(6))
    ).to_pure_dict()
    right_branch = nnx.state(
        _residual_action_branch()(depth=4, width=64, input_width=128, rngs=nnx.Rngs(7))
    ).to_pure_dict()

    def objective(gate):
        branch = {**left_branch, "gate": gate}
        outputs, _, _ = paired_gemma_forward(
            left_params,
            right_params,
            configs,
            left_inputs,
            right_inputs,
            remote_action_kv_enabled=False,
            remote_action_residual_enabled=True,
            left_residual_params=branch,
            right_residual_params=right_branch,
            **kwargs,
        )
        return jnp.sum(outputs[1])

    zero = jnp.zeros((4,), dtype=jnp.float32)
    gradient = jax.grad(objective)(zero)
    assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.linalg.norm(gradient)) > 0
    assert float(jnp.abs(objective(jnp.full((4,), 0.1)) - objective(zero))) > 0
