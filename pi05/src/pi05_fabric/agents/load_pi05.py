"""Load independent pi0.5 LoRA graphs from the immutable base checkpoint."""

from __future__ import annotations

from pathlib import Path

from flax import nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

from openpi.models import pi0_config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training import weight_loaders

from pi05_fabric.agents.dual_pi05 import DualPi05
from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.agents.pi05_lora import project_trainable_filter
from pi05_fabric.agents.pi05_strong import native_trainable_filter


def pi05_lora_config(*, action_horizon: int = 20) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=action_horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )


def _load_partial_params(checkpoint: Path, params_shape):
    loaded = weight_loaders.CheckpointWeightLoader(str(checkpoint / "params")).load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


def load_single_pi05(
    checkpoint: str | Path,
    *,
    key: jax.Array,
    trainable_filter=None,
    action_horizon: int = 20,
):
    checkpoint = Path(checkpoint)
    config = pi05_lora_config(action_horizon=action_horizon)
    shaped_model = nnx.eval_shape(config.create, key)
    graphdef, shaped_state = nnx.split(shaped_model)
    partial = _load_partial_params(checkpoint, shaped_state.to_pure_dict())

    def initialize(rng, loaded):
        model = config.create(rng)
        _, state = nnx.split(model)
        state.replace_by_pure_dict(loaded)
        selected_filter = (
            project_trainable_filter(include_linear_action=False)
            if trainable_filter is None
            else trainable_filter
        )
        frozen = nnx.All(nnx.Param, nnx.Not(selected_filter))
        return nnx_utils.state_map(
            state,
            frozen,
            lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )

    state = jax.jit(initialize, donate_argnums=(1,))(key, partial)
    jax.block_until_ready(state)
    return config, nnx.merge(graphdef, state)


def load_dual_pi05(
    checkpoint: str | Path,
    *,
    seed: int = 0,
    mode: DualPi05Mode | None = None,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
    train_paligemma_qo: bool = False,
    separate_expanded_groups: bool = False,
    action_horizon: int = 20,
) -> tuple[pi0_config.Pi0Config, DualPi05]:
    left_key, right_key, fabric_key = jax.random.split(jax.random.key(seed), 3)
    trainable_filter = (
        native_trainable_filter(
            train_action_ffw=train_action_ffw,
            train_action_attention=train_action_attention,
            train_paligemma_kv=train_paligemma_kv,
            train_paligemma_qo=train_paligemma_qo,
            separate_expanded_groups=separate_expanded_groups,
        )
        if mode is not None and mode.is_native
        else None
    )
    config, left = load_single_pi05(
        checkpoint, key=left_key, trainable_filter=trainable_filter, action_horizon=action_horizon
    )
    right_config, right = load_single_pi05(
        checkpoint, key=right_key, trainable_filter=trainable_filter, action_horizon=action_horizon
    )
    if config != right_config:
        raise RuntimeError("left and right pi0.5 configurations differ")
    return config, DualPi05(left, right, mode=mode, rngs=nnx.Rngs(fabric_key))
