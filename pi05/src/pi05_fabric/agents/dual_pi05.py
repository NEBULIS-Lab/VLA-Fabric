"""Two complete pi0.5 policies connected by the selected RAW interface."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import model as model_api
from openpi.models import pi0

from pi05_fabric.communication.paired_gemma import PairedGemmaTrace
from pi05_fabric.communication.paired_gemma import paired_gemma_forward
from pi05_fabric.communication.interaction_spec import InteractionSpec
from pi05_fabric.communication.residual_action import ResidualActionBranch


class DualPi05Mode(str, Enum):
    INDEPENDENT = "independent"
    RAW_COMMON_ONLY = "raw_common_only"
    I3_RAW_CORE = "i3_raw_core"
    I4_RAW_FULL = "i4_raw_full"
    PI_NATIVE_INDEPENDENT = "pi_native_independent"
    PI_NATIVE_RAW_COMMON_ONLY = "pi_native_raw_common_only"
    PI_NATIVE_CORE = "pi_native_core"
    PI_NATIVE_THREE_PATH = "pi_native_three_path"
    PI_NATIVE_V2_INDEPENDENT = "pi_native_v2_independent"
    PI_NATIVE_V2_RAW_COMMON_ONLY = "pi_native_v2_raw_common_only"
    PI_NATIVE_V2_CORE = "pi_native_v2_core"
    PI_NATIVE_V2_WO_COMMON = "pi_native_v2_wo_common"
    PI_NATIVE_V2_WO_PRIVATE = "pi_native_v2_wo_private"
    PI_NATIVE_V2_ALL_OFF = "pi_native_v2_all_off"
    PI_NATIVE_V2_RESIDUAL_ACTION = "pi_native_v2_residual_action"

    @property
    def interaction_spec(self) -> InteractionSpec:
        return {
            self.INDEPENDENT: InteractionSpec(False, False, False, False),
            self.RAW_COMMON_ONLY: InteractionSpec(True, False, False, False),
            self.I3_RAW_CORE: InteractionSpec(True, True, False, False),
            self.I4_RAW_FULL: InteractionSpec(True, True, False, True),
            self.PI_NATIVE_INDEPENDENT: InteractionSpec(False, False, False, False),
            self.PI_NATIVE_RAW_COMMON_ONLY: InteractionSpec(True, False, False, False),
            self.PI_NATIVE_CORE: InteractionSpec(True, True, False, False),
            self.PI_NATIVE_THREE_PATH: InteractionSpec(True, True, True, False),
            self.PI_NATIVE_V2_INDEPENDENT: InteractionSpec(False, False, False, False),
            self.PI_NATIVE_V2_RAW_COMMON_ONLY: InteractionSpec(True, False, False, False),
            self.PI_NATIVE_V2_CORE: InteractionSpec(True, True, False, False),
            self.PI_NATIVE_V2_WO_COMMON: InteractionSpec(False, True, False, False),
            self.PI_NATIVE_V2_WO_PRIVATE: InteractionSpec(True, False, False, False),
            self.PI_NATIVE_V2_ALL_OFF: InteractionSpec(False, False, False, False),
            self.PI_NATIVE_V2_RESIDUAL_ACTION: InteractionSpec(True, True, False, False),
        }[self]

    @property
    def remote_action_residual(self) -> bool:
        return self in {
            self.PI_NATIVE_V2_WO_COMMON,
            self.PI_NATIVE_V2_WO_PRIVATE,
            self.PI_NATIVE_V2_RESIDUAL_ACTION,
        }

    @property
    def instantiates_residual_action(self) -> bool:
        return self in {
            self.PI_NATIVE_V2_RAW_COMMON_ONLY,
            self.PI_NATIVE_V2_CORE,
            self.PI_NATIVE_V2_WO_COMMON,
            self.PI_NATIVE_V2_WO_PRIVATE,
            self.PI_NATIVE_V2_ALL_OFF,
            self.PI_NATIVE_V2_RESIDUAL_ACTION,
        }


    @property
    def is_v2(self) -> bool:
        return self in {
            self.PI_NATIVE_V2_INDEPENDENT,
            self.PI_NATIVE_V2_RAW_COMMON_ONLY,
            self.PI_NATIVE_V2_CORE,
            self.PI_NATIVE_V2_WO_COMMON,
            self.PI_NATIVE_V2_WO_PRIVATE,
            self.PI_NATIVE_V2_ALL_OFF,
            self.PI_NATIVE_V2_RESIDUAL_ACTION,
        }

    @property
    def is_native(self) -> bool:
        return self in {
            self.PI_NATIVE_INDEPENDENT,
            self.PI_NATIVE_V2_INDEPENDENT,
            self.PI_NATIVE_RAW_COMMON_ONLY,
            self.PI_NATIVE_CORE,
            self.PI_NATIVE_THREE_PATH,
            self.PI_NATIVE_V2_RAW_COMMON_ONLY,
            self.PI_NATIVE_V2_CORE,
            self.PI_NATIVE_V2_WO_COMMON,
            self.PI_NATIVE_V2_WO_PRIVATE,
            self.PI_NATIVE_V2_ALL_OFF,
            self.PI_NATIVE_V2_RESIDUAL_ACTION,
        }

class DualPi05Loss(NamedTuple):
    total: jax.Array
    left: jax.Array
    right: jax.Array
    trace: PairedGemmaTrace


def _local_identity_kernel(_key, shape, dtype=jnp.float32):
    if len(shape) != 2 or shape[0] != 2 * shape[1]:
        raise ValueError("Linear Action kernel must have shape [2D, D]")
    width = shape[1]
    return jnp.concatenate(
        [jnp.eye(width, dtype=dtype), jnp.zeros((width, width), dtype=dtype)],
        axis=0,
    )


class DualPi05(nnx.Module):
    """Own two independent models and expose one summed dual-agent loss."""

    def __init__(
        self,
        left: pi0.Pi0,
        right: pi0.Pi0,
        *,
        mode: DualPi05Mode | None = None,
        rngs: nnx.Rngs,
    ):
        if left is right:
            raise ValueError("left and right must be independent pi0.5 instances")
        self.left = left
        self.right = right
        width = left.action_in_proj.out_features
        if right.action_in_proj.out_features != width:
            raise ValueError("both action experts must use the same hidden width")
        if mode is not None and mode.instantiates_residual_action:
            depth = left.PaliGemma.llm.module.configs[0].depth
            action_config = left.PaliGemma.llm.module.configs[1]
            residual_input_width = action_config.num_heads * action_config.head_dim
            self.fabric_residual_action_left = ResidualActionBranch(
                depth=depth, width=width, input_width=residual_input_width, rngs=rngs
            )
            self.fabric_residual_action_right = ResidualActionBranch(
                depth=depth, width=width, input_width=residual_input_width, rngs=rngs
            )
        if mode is None or not mode.is_native:
            self.fabric_linear_action_left = nnx.Linear(
                2 * width,
                width,
                kernel_init=_local_identity_kernel,
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
            self.fabric_linear_action_right = nnx.Linear(
                2 * width,
                width,
                kernel_init=_local_identity_kernel,
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
        self.gemma_configs = tuple(left.PaliGemma.llm.module.configs)

    def _residual_action_params(self, mode: DualPi05Mode):
        if not mode.remote_action_residual:
            return None, None
        return (
            nnx.state(self.fabric_residual_action_left).to_pure_dict(),
            nnx.state(self.fabric_residual_action_right).to_pure_dict(),
        )

    def compute_loss(
        self,
        rng,
        left_observation,
        right_observation,
        left_actions,
        right_actions,
        *,
        ownership,
        mode: DualPi05Mode,
        train: bool,
        preprocessed: bool = False,
    ) -> DualPi05Loss:
        left_preprocess, right_preprocess, left_noise_key, right_noise_key, time_key = jax.random.split(rng, 5)
        if not preprocessed:
            left_observation = model_api.preprocess_observation(left_preprocess, left_observation, train=train)
            right_observation = model_api.preprocess_observation(right_preprocess, right_observation, train=train)
        if left_actions.shape != right_actions.shape:
            raise ValueError("left and right action tensors must have the same shape")

        batch_shape = left_actions.shape[:-2]
        left_noise = jax.random.normal(left_noise_key, left_actions.shape)
        right_noise = jax.random.normal(right_noise_key, right_actions.shape)
        time = jax.random.beta(time_key, 1.5, 1.0, batch_shape) * 0.999 + 0.001
        expanded_time = time[..., None, None]
        left_xt = expanded_time * left_noise + (1 - expanded_time) * left_actions
        right_xt = expanded_time * right_noise + (1 - expanded_time) * right_actions
        left_target, right_target = left_noise - left_actions, right_noise - right_actions

        left_prefix, left_prefix_mask, left_prefix_ar = self.left.embed_prefix(left_observation)
        right_prefix, right_prefix_mask, right_prefix_ar = self.right.embed_prefix(right_observation)
        left_suffix, left_suffix_mask, left_suffix_ar, left_cond = self.left.embed_suffix(left_observation, left_xt, time)
        right_suffix, right_suffix_mask, right_suffix_ar, right_cond = self.right.embed_suffix(right_observation, right_xt, time)
        left_input_mask = jnp.concatenate([left_prefix_mask, left_suffix_mask], axis=1)
        right_input_mask = jnp.concatenate([right_prefix_mask, right_suffix_mask], axis=1)
        left_ar = jnp.concatenate([left_prefix_ar, left_suffix_ar], axis=0)
        right_ar = jnp.concatenate([right_prefix_ar, right_suffix_ar], axis=0)
        left_mask = pi0.make_attn_mask(left_input_mask, left_ar)
        right_mask = pi0.make_attn_mask(right_input_mask, right_ar)
        left_positions = jnp.cumsum(left_input_mask, axis=1) - 1
        right_positions = jnp.cumsum(right_input_mask, axis=1) - 1

        interaction_spec = mode.interaction_spec
        left_residual_params, right_residual_params = self._residual_action_params(mode)
        left_out, right_out, trace = paired_gemma_forward(
            nnx.state(self.left.PaliGemma.llm).to_pure_dict(),
            nnx.state(self.right.PaliGemma.llm).to_pure_dict(),
            self.gemma_configs,
            (left_prefix, left_suffix),
            (right_prefix, right_suffix),
            left_positions=left_positions,
            right_positions=right_positions,
            left_mask=left_mask,
            right_mask=right_mask,
            left_adarms_cond=(None, left_cond),
            right_adarms_cond=(None, right_cond),
            ownership=ownership,
            common_enabled=interaction_spec.common,
            private_kv_enabled=interaction_spec.private_kv,
            remote_action_kv_enabled=interaction_spec.remote_action_kv,
            action_horizon=self.left.action_horizon,
            remote_action_residual_enabled=mode.remote_action_residual,
            left_residual_params=left_residual_params,
            right_residual_params=right_residual_params,
        )
        left_hidden = left_out[1][:, -self.left.action_horizon :]
        right_hidden = right_out[1][:, -self.right.action_horizon :]
        if interaction_spec.final_linear_action:
            original_left, original_right = left_hidden, right_hidden
            left_hidden = self.fabric_linear_action_left(jnp.concatenate([original_left, original_right], axis=-1))
            right_hidden = self.fabric_linear_action_right(jnp.concatenate([original_right, original_left], axis=-1))
        left_velocity = self.left.action_out_proj(left_hidden)
        right_velocity = self.right.action_out_proj(right_hidden)
        left_loss = jnp.mean(jnp.square(left_velocity - left_target), axis=-1)
        right_loss = jnp.mean(jnp.square(right_velocity - right_target), axis=-1)
        return DualPi05Loss(jnp.mean(left_loss) + jnp.mean(right_loss), left_loss, right_loss, trace)


    def sample_actions(
        self,
        rng,
        left_observation,
        right_observation,
        *,
        ownership,
        mode: DualPi05Mode,
        num_steps: int = 10,
        left_noise=None,
        right_noise=None,
        preprocessed: bool = False,
    ):
        """Sample one local action chunk per agent through the paired RAW graph."""

        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not preprocessed:
            left_observation = model_api.preprocess_observation(None, left_observation, train=False)
            right_observation = model_api.preprocess_observation(None, right_observation, train=False)
        left_key, right_key = jax.random.split(rng)
        batch_size = left_observation.state.shape[0]
        if right_observation.state.shape[0] != batch_size:
            raise ValueError("left and right observations must have equal batch sizes")
        action_shape = (batch_size, self.left.action_horizon, self.left.action_dim)
        left_noise = jax.random.normal(left_key, action_shape) if left_noise is None else left_noise
        right_noise = jax.random.normal(right_key, action_shape) if right_noise is None else right_noise

        left_prefix, left_prefix_mask, left_prefix_ar = self.left.embed_prefix(left_observation)
        right_prefix, right_prefix_mask, right_prefix_ar = self.right.embed_prefix(right_observation)
        interaction_spec = mode.interaction_spec
        left_residual_params, right_residual_params = self._residual_action_params(mode)
        dt = -1.0 / num_steps

        def denoise_step(_, carry):
            left_x, right_x, time = carry
            times = jnp.broadcast_to(time, (batch_size,))
            left_suffix, left_suffix_mask, left_suffix_ar, left_cond = self.left.embed_suffix(
                left_observation, left_x, times
            )
            right_suffix, right_suffix_mask, right_suffix_ar, right_cond = self.right.embed_suffix(
                right_observation, right_x, times
            )
            left_input_mask = jnp.concatenate([left_prefix_mask, left_suffix_mask], axis=1)
            right_input_mask = jnp.concatenate([right_prefix_mask, right_suffix_mask], axis=1)
            left_ar = jnp.concatenate([left_prefix_ar, left_suffix_ar], axis=0)
            right_ar = jnp.concatenate([right_prefix_ar, right_suffix_ar], axis=0)
            left_mask = pi0.make_attn_mask(left_input_mask, left_ar)
            right_mask = pi0.make_attn_mask(right_input_mask, right_ar)
            left_positions = jnp.cumsum(left_input_mask, axis=1) - 1
            right_positions = jnp.cumsum(right_input_mask, axis=1) - 1
            left_out, right_out, _ = paired_gemma_forward(
                nnx.state(self.left.PaliGemma.llm).to_pure_dict(),
                nnx.state(self.right.PaliGemma.llm).to_pure_dict(),
                self.gemma_configs,
                (left_prefix, left_suffix),
                (right_prefix, right_suffix),
                left_positions=left_positions,
                right_positions=right_positions,
                left_mask=left_mask,
                right_mask=right_mask,
                left_adarms_cond=(None, left_cond),
                right_adarms_cond=(None, right_cond),
                ownership=ownership,
                common_enabled=interaction_spec.common,
                private_kv_enabled=interaction_spec.private_kv,
                remote_action_kv_enabled=interaction_spec.remote_action_kv,
                action_horizon=self.left.action_horizon,
                remote_action_residual_enabled=mode.remote_action_residual,
                left_residual_params=left_residual_params,
                right_residual_params=right_residual_params,
            )
            left_hidden = left_out[1][:, -self.left.action_horizon :]
            right_hidden = right_out[1][:, -self.right.action_horizon :]
            if interaction_spec.final_linear_action:
                original_left, original_right = left_hidden, right_hidden
                left_hidden = self.fabric_linear_action_left(
                    jnp.concatenate([original_left, original_right], axis=-1)
                )
                right_hidden = self.fabric_linear_action_right(
                    jnp.concatenate([original_right, original_left], axis=-1)
                )
            left_velocity = self.left.action_out_proj(left_hidden)
            right_velocity = self.right.action_out_proj(right_hidden)
            return left_x + dt * left_velocity, right_x + dt * right_velocity, time + dt

        left_actions, right_actions, _ = jax.lax.fori_loop(
            0,
            num_steps,
            denoise_step,
            (left_noise, right_noise, jnp.asarray(1.0, dtype=jnp.float32)),
        )
        return left_actions, right_actions
