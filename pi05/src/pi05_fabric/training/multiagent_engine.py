"""One-role-per-process training engine for multi-agent PI0.5."""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pi05_fabric.agents.multiagent_pi05 import MultiAgentPi05Mode
from pi05_fabric.agents.pi05_strong import native_optimizer_labels
from pi05_fabric.agents.pi05_strong import native_trainable_filter
from pi05_fabric.training.agent_parallel import AgentParallelTopology
from pi05_fabric.training.distributed import shard_batch
from pi05_fabric.training.optimizer import GroupedOptimizerSettings
from pi05_fabric.training.optimizer import create_grouped_optimizer


class MultiAgentEngineState(NamedTuple):
    model_state: Any
    opt_state: Any
    step: jax.Array


class MultiAgentStepInfo(NamedTuple):
    team_loss: jax.Array
    local_loss: jax.Array
    role_losses: jax.Array
    grad_norm: jax.Array
    local_grad_norm: jax.Array
    finite: jax.Array


def finite_gate_reference(values) -> bool:
    values = np.asarray(values)
    return bool(values.size and np.all(values))


class MultiAgentTrainingEngine:
    def __init__(
        self,
        model,
        *,
        mode: MultiAgentPi05Mode,
        topology: AgentParallelTopology,
        tx: optax.GradientTransformation,
        trainable_filter=None,
    ):
        self.mode = mode
        self.topology = topology
        self.trainable_filter = trainable_filter or native_trainable_filter(
            train_action_ffw=False,
            train_action_attention=True,
            train_paligemma_kv=True,
        )
        self.graphdef, model_state = nnx.split(model)
        trainable = model_state.filter(self.trainable_filter)
        self.frozen_state = model_state.filter(nnx.Not(self.trainable_filter))
        self.tx = tx
        self.state = MultiAgentEngineState(
            trainable,
            tx.init(trainable),
            jnp.asarray(0, dtype=jnp.int32),
        )
        self._parallel_steps = {}

    def current_step(self) -> int:
        return int(self.state.step)

    def selected_params(self):
        return self.state.model_state

    def full_model_state(self):
        return nnx.State.merge(self.frozen_state, self.state.model_state)

    def _loss_and_grads(
        self,
        state,
        frozen_state,
        rng,
        observation,
        actions,
        ownership,
        *,
        preprocessed,
        axis_name,
    ):
        model = nnx.merge(
            self.graphdef,
            nnx.State.merge(frozen_state, state.model_state),
        )

        def loss_fn(current_model):
            result = current_model.compute_loss(
                rng,
                observation,
                actions,
                ownership=ownership,
                mode=self.mode,
                axis_name=axis_name,
                agent_groups=self.topology.agent_groups,
                train=True,
                preprocessed=preprocessed,
            )
            return result.total, jnp.mean(result.local)

        diff_state = nnx.DiffState(0, self.trainable_filter)
        (team_loss, local_loss), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model)
        return team_loss, local_loss, grads

    def _finish_step(
        self,
        state,
        grads,
        team_loss,
        local_loss,
        local_finite,
        *,
        axis_name,
    ):
        local_grad_norm = optax.global_norm(grads)
        grads = jax.lax.pmean(
            grads,
            axis_name,
            axis_index_groups=self.topology.role_data_parallel_groups,
        )
        finite = jax.lax.pmin(
            local_finite.astype(jnp.int32),
            axis_name,
            axis_index_groups=self.topology.agent_groups,
        )
        finite = jax.lax.pmin(
            finite,
            axis_name,
            axis_index_groups=self.topology.role_data_parallel_groups,
        )
        finite = jax.lax.pmin(
            finite,
            axis_name,
            axis_index_groups=self.topology.agent_groups,
        ).astype(bool)
        role_losses = jax.lax.all_gather(
            local_loss,
            axis_name,
            axis=0,
            tiled=False,
            axis_index_groups=self.topology.agent_groups,
        )
        grad_norm = optax.global_norm(grads)
        updates, next_opt_state = self.tx.update(
            grads,
            state.opt_state,
            state.model_state,
        )
        updates = jax.tree.map(
            lambda update: jnp.where(finite, update, jnp.zeros_like(update)),
            updates,
        )
        next_opt_state = jax.tree.map(
            lambda new, old: jnp.where(finite, new, old),
            next_opt_state,
            state.opt_state,
        )
        next_params = optax.apply_updates(state.model_state, updates)
        next_state = MultiAgentEngineState(
            next_params,
            next_opt_state,
            state.step + finite.astype(jnp.int32),
        )
        return next_state, MultiAgentStepInfo(
            team_loss,
            local_loss,
            role_losses,
            grad_norm,
            local_grad_norm,
            finite,
        )

    def _step_impl(
        self,
        state,
        frozen_state,
        rng,
        observation,
        actions,
        ownership,
        *,
        preprocessed,
        gradient_accumulation,
        axis_name,
    ):
        def split_microbatches(value):
            if value.shape[0] % gradient_accumulation:
                raise ValueError(
                    "local team batch is not divisible by gradient accumulation"
                )
            microbatch = value.shape[0] // gradient_accumulation
            return value.reshape(
                (gradient_accumulation, microbatch, *value.shape[1:])
            )

        observations = jax.tree.map(split_microbatches, observation)
        action_batches = split_microbatches(actions)
        ownership_batches = split_microbatches(ownership)
        keys = jax.random.split(rng, gradient_accumulation)

        def evaluate(index):
            return self._loss_and_grads(
                state,
                frozen_state,
                keys[index],
                jax.tree.map(lambda value: value[index], observations),
                action_batches[index],
                ownership_batches[index],
                preprocessed=preprocessed,
                axis_name=axis_name,
            )

        first_team, first_local, first_grads = evaluate(0)
        grad_sum = jax.tree.map(lambda value: value.astype(jnp.float32), first_grads)
        finite = jnp.all(
            jnp.asarray(
                [
                    jnp.isfinite(first_team),
                    jnp.isfinite(first_local),
                    jnp.isfinite(optax.global_norm(first_grads)),
                ]
            )
        )

        def accumulate(carry, index):
            grads_total, team_total, local_total, all_finite = carry
            team_loss, local_loss, grads = evaluate(index)
            grads_total = jax.tree.map(
                lambda total, value: total + value.astype(jnp.float32),
                grads_total,
                grads,
            )
            current_finite = jnp.all(
                jnp.asarray(
                    [
                        jnp.isfinite(team_loss),
                        jnp.isfinite(local_loss),
                        jnp.isfinite(optax.global_norm(grads)),
                    ]
                )
            )
            return (
                grads_total,
                team_total + team_loss,
                local_total + local_loss,
                jnp.logical_and(all_finite, current_finite),
            ), None

        (grad_sum, team_sum, local_sum, finite), _ = jax.lax.scan(
            accumulate,
            (grad_sum, first_team, first_local, finite),
            jnp.arange(1, gradient_accumulation),
        )
        scale = jnp.asarray(gradient_accumulation, dtype=jnp.float32)
        grads = jax.tree.map(lambda value: value / scale, grad_sum)
        return self._finish_step(
            state,
            grads,
            team_sum / scale,
            local_sum / scale,
            finite,
            axis_name=axis_name,
        )

    def step(
        self,
        rng,
        observation,
        actions,
        ownership,
        *,
        preprocessed: bool,
        gradient_accumulation: int,
    ):
        if gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        if jax.process_count() != self.topology.world_size:
            raise ValueError(
                "JAX process count does not match the declared agent topology"
            )
        if jax.local_device_count() != 1:
            raise ValueError("agent-parallel training requires one local GPU per process")
        key = (preprocessed, gradient_accumulation)
        if key not in self._parallel_steps:
            def parallel_step(state, frozen, local_rng, obs, act, owner):
                return self._step_impl(
                    state,
                    frozen,
                    local_rng,
                    obs,
                    act,
                    owner,
                    preprocessed=preprocessed,
                    gradient_accumulation=gradient_accumulation,
                    axis_name="agents",
                )

            self._parallel_steps[key] = jax.pmap(
                parallel_step,
                axis_name="agents",
                in_axes=(None, None, 0, 0, 0, 0),
            )
        rngs = jnp.expand_dims(rng, axis=0)
        observation, actions, ownership = shard_batch(
            (observation, actions, ownership),
            device_count=1,
        )
        next_state, info = self._parallel_steps[key](
            self.state,
            self.frozen_state,
            rngs,
            observation,
            actions,
            ownership,
        )
        self.state = jax.tree.map(lambda value: value[0], next_state)
        return info


def create_multiagent_engine(
    model,
    *,
    mode: MultiAgentPi05Mode,
    topology: AgentParallelTopology,
    settings: GroupedOptimizerSettings,
) -> MultiAgentTrainingEngine:
    trainable_filter = native_trainable_filter(
        train_action_ffw=False,
        train_action_attention=True,
        train_paligemma_kv=True,
    )
    model_state = nnx.state(model)
    trainable = model_state.filter(trainable_filter)
    labels = native_optimizer_labels(
        trainable,
        train_action_ffw=False,
        train_action_attention=True,
        train_paligemma_kv=True,
    )
    tx = create_grouped_optimizer(settings, labels)
    return MultiAgentTrainingEngine(
        model,
        mode=mode,
        topology=topology,
        tx=tx,
        trainable_filter=trainable_filter,
    )
