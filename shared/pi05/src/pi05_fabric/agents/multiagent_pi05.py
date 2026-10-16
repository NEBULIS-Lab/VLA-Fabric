"""Local role model contracts for agent-parallel PI0.5 teams."""

from __future__ import annotations

import math
from enum import Enum
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_api
from openpi.models import pi0

from pi05_fabric.communication.multiagent_gemma import MultiAgentGemmaTrace
from pi05_fabric.communication.multiagent_gemma import multiagent_gemma_forward


class MultiAgentPi05Mode(str, Enum):
    INDEPENDENT = "independent"
    COMMON_ONLY = "common_only"
    CORE = "core"
    FULL = "full"

    @property
    def common(self) -> bool:
        return self is not self.INDEPENDENT

    @property
    def private_kv(self) -> bool:
        return self in {self.CORE, self.FULL}

    @property
    def residual_action(self) -> bool:
        return self is self.FULL


class RoleRngs(NamedTuple):
    preprocess: jax.Array
    noise: jax.Array
    time: jax.Array


def fold_role_rng(key, *, role_index: int) -> RoleRngs:
    if isinstance(role_index, int) and role_index < 0:
        raise ValueError("role_index must be non-negative")
    return RoleRngs(
        preprocess=jax.random.fold_in(key, 100 + role_index),
        noise=jax.random.fold_in(key, 200 + role_index),
        time=jax.random.fold_in(key, 300),
    )


def team_loss_reference(role_losses):
    role_losses = jnp.asarray(role_losses)
    if role_losses.ndim != 1 or role_losses.size < 2:
        raise ValueError("role_losses must contain one scalar per agent")
    return jnp.mean(role_losses), role_losses


class MultiAgentResidualActionBranch(nnx.Module):
    """Receiver-local projection over role-ordered peer Action contexts."""

    def __init__(
        self,
        *,
        depth: int,
        width: int,
        peer_width: int,
        max_agents: int,
        rngs: nnx.Rngs,
    ):
        if depth <= 0 or width <= 0 or peer_width <= 0 or max_agents < 2:
            raise ValueError("residual action dimensions must be positive")
        input_width = (max_agents - 1) * peer_width
        kernel = jax.random.normal(
            rngs.params(), (depth, input_width, width), dtype=jnp.float32
        ) / math.sqrt(input_width)
        self.kernel = nnx.Param(kernel)
        self.gate = nnx.Param(jnp.zeros((depth,), dtype=jnp.float32))


class MultiAgentPi05Loss(NamedTuple):
    total: jax.Array
    local: jax.Array
    trace: MultiAgentGemmaTrace


class LocalRolePi05(nnx.Module):
    """One complete role-local PI0.5 policy with explicit agent collectives."""

    def __init__(
        self,
        policy: pi0.Pi0,
        *,
        agent_count: int,
        max_agents: int,
        rngs: nnx.Rngs,
    ):
        if not 2 <= agent_count <= max_agents:
            raise ValueError("agent_count must be between two and max_agents")
        self.policy = policy
        self.agent_count = agent_count
        self.max_agents = max_agents
        self.gemma_configs = tuple(policy.PaliGemma.llm.module.configs)
        action_config = self.gemma_configs[1]
        peer_width = action_config.num_heads * action_config.head_dim
        self.residual_action = MultiAgentResidualActionBranch(
            depth=self.gemma_configs[0].depth,
            width=policy.action_in_proj.out_features,
            peer_width=peer_width,
            max_agents=max_agents,
            rngs=rngs,
        )

    def _forward(
        self,
        prefix,
        suffix,
        *,
        positions,
        mask,
        cond,
        ownership,
        mode: MultiAgentPi05Mode,
        axis_name: str,
        agent_groups,
    ):
        residual_params = None
        if mode.residual_action:
            residual_params = nnx.state(self.residual_action).to_pure_dict()
        return multiagent_gemma_forward(
            nnx.state(self.policy.PaliGemma.llm).to_pure_dict(),
            self.gemma_configs,
            (prefix, suffix),
            positions=positions,
            mask=mask,
            adarms_cond=(None, cond),
            ownership=ownership,
            agent_count=self.agent_count,
            max_agents=self.max_agents,
            axis_name=axis_name,
            agent_groups=agent_groups,
            common_enabled=mode.common,
            private_kv_enabled=mode.private_kv,
            residual_action_enabled=mode.residual_action,
            action_horizon=self.policy.action_horizon,
            residual_params=residual_params,
        )

    def compute_loss(
        self,
        rng,
        observation,
        actions,
        *,
        ownership,
        mode: MultiAgentPi05Mode,
        axis_name: str,
        agent_groups,
        train: bool,
        preprocessed: bool = False,
    ) -> MultiAgentPi05Loss:
        role = jax.lax.axis_index(axis_name) % self.agent_count
        role_rngs = fold_role_rng(rng, role_index=role)
        if not preprocessed:
            observation = model_api.preprocess_observation(
                role_rngs.preprocess,
                observation,
                train=train,
            )
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(role_rngs.noise, actions.shape)
        time = jax.random.beta(role_rngs.time, 1.5, 1.0, batch_shape) * 0.999 + 0.001
        expanded_time = time[..., None, None]
        noisy_actions = expanded_time * noise + (1 - expanded_time) * actions
        target = noise - actions

        prefix, prefix_mask, prefix_ar = self.policy.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, cond = self.policy.embed_suffix(
            observation,
            noisy_actions,
            time,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attention_mask = pi0.make_attn_mask(input_mask, ar)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        output, trace = self._forward(
            prefix,
            suffix,
            positions=positions,
            mask=attention_mask,
            cond=cond,
            ownership=ownership,
            mode=mode,
            axis_name=axis_name,
            agent_groups=agent_groups,
        )
        hidden = output[1][:, -self.policy.action_horizon :]
        velocity = self.policy.action_out_proj(hidden)
        local_loss = jnp.mean(jnp.square(velocity - target), axis=-1)
        local_mean = jnp.mean(local_loss)
        team_total = jax.lax.psum(
            local_mean,
            axis_name,
            axis_index_groups=agent_groups,
        ) / self.agent_count
        return MultiAgentPi05Loss(team_total, local_loss, trace)

    def sample_actions(
        self,
        rng,
        observation,
        *,
        ownership,
        mode: MultiAgentPi05Mode,
        axis_name: str,
        agent_groups,
        num_steps: int = 10,
        noise=None,
        preprocessed: bool = False,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        role = jax.lax.axis_index(axis_name) % self.agent_count
        role_rngs = fold_role_rng(rng, role_index=role)
        if not preprocessed:
            observation = model_api.preprocess_observation(
                role_rngs.preprocess,
                observation,
                train=False,
            )
        batch_size = observation.state.shape[0]
        action_shape = (
            batch_size,
            self.policy.action_horizon,
            self.policy.action_dim,
        )
        actions = (
            jax.random.normal(role_rngs.noise, action_shape)
            if noise is None
            else noise
        )
        prefix, prefix_mask, prefix_ar = self.policy.embed_prefix(observation)
        dt = -1.0 / num_steps

        def denoise_step(_, carry):
            current_actions, current_time = carry
            times = jnp.broadcast_to(current_time, (batch_size,))
            suffix, suffix_mask, suffix_ar, cond = self.policy.embed_suffix(
                observation,
                current_actions,
                times,
            )
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
            attention_mask = pi0.make_attn_mask(input_mask, ar)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            output, _ = self._forward(
                prefix,
                suffix,
                positions=positions,
                mask=attention_mask,
                cond=cond,
                ownership=ownership,
                mode=mode,
                axis_name=axis_name,
                agent_groups=agent_groups,
            )
            hidden = output[1][:, -self.policy.action_horizon :]
            velocity = self.policy.action_out_proj(hidden)
            return current_actions + dt * velocity, current_time + dt

        actions, _ = jax.lax.fori_loop(
            0,
            num_steps,
            denoise_step,
            (actions, jnp.asarray(1.0, dtype=jnp.float32)),
        )
        return actions
