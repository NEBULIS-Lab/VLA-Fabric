"""JIT-compiled LoRA training step for the paired pi0.5 model."""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.agents.pi05_lora import project_trainable_filter
from pi05_fabric.agents.pi05_strong import native_optimizer_labels
from pi05_fabric.agents.pi05_strong import native_trainable_filter
from pi05_fabric.training.distributed import process_device_rngs
from pi05_fabric.training.distributed import shard_batch
from pi05_fabric.training.distributed import unreplicate_tree


class EngineState(NamedTuple):
    model_state: Any
    opt_state: Any
    step: jax.Array


class StepInfo(NamedTuple):
    loss: jax.Array
    left_loss: jax.Array
    right_loss: jax.Array
    grad_norm: jax.Array
    finite: jax.Array
    local_loss: jax.Array
    local_left_loss: jax.Array
    local_right_loss: jax.Array
    local_grad_norm: jax.Array


def parameter_values(state) -> dict[str, jax.Array]:
    return {
        "/".join(str(part) for part in path): value.value
        for path, value in state.flat_state().items()
    }


class TrainingEngine:
    def __init__(
        self,
        model,
        *,
        mode: DualPi05Mode,
        tx: optax.GradientTransformation,
        trainable_filter=None,
    ):
        self.mode = mode
        self.trainable_filter = (
            project_trainable_filter(include_linear_action=mode is DualPi05Mode.I4_RAW_FULL)
            if trainable_filter is None
            else trainable_filter
        )
        self.graphdef, model_state = nnx.split(model)
        trainable = model_state.filter(self.trainable_filter)
        self.frozen_state = model_state.filter(nnx.Not(self.trainable_filter))
        self.tx = tx
        # Only mutable parameters belong to the optimizer state. Keeping the
        # frozen base model here would make pmap return both complete pi0.5
        # parameter trees after every update and require a second large buffer.
        self.state = EngineState(trainable, tx.init(trainable), jnp.asarray(0, dtype=jnp.int32))
        self._distributed_devices = ()
        self._state_has_replica_axis = False
        self._parallel_steps = {}
        self._compiled_step = jax.jit(
            partial(self._step_impl),
            static_argnames=("preprocessed",),
        )
        self._compiled_accumulated_step = jax.jit(
            partial(self._accumulated_step_impl),
            static_argnames=("preprocessed", "gradient_accumulation"),
        )

    @property
    def device_count(self) -> int:
        return len(self._distributed_devices) or 1

    @property
    def distributed(self) -> bool:
        return bool(self._distributed_devices)

    def enable_data_parallel(self, devices) -> None:
        devices = tuple(devices)
        if not devices:
            raise ValueError("data parallel training requires a local device")
        if self.distributed:
            raise RuntimeError("data parallel training is already enabled")
        # A multi-process launch already owns one complete state per process.
        # Adding a local replica axis in that topology creates a second copy on
        # the same GPU before the first update. Local multi-device launches do
        # need explicit replication because one process owns every device.
        if jax.process_count() == 1:
            self.state = jax.device_put_replicated(self.state, devices)
            self._state_has_replica_axis = True
        self._distributed_devices = devices

    def host_state(self, state: EngineState | None = None) -> EngineState:
        state = self.state if state is None else state
        return unreplicate_tree(state) if self._state_has_replica_axis else state

    def current_step(self) -> int:
        return int(self.host_state().step)

    def selected_params(self, state: EngineState | None = None):
        state = self.state if state is None else state
        return state.model_state

    def full_model_state(self, state: EngineState | None = None):
        state = self.state if state is None else state
        return nnx.State.merge(self.frozen_state, state.model_state)

    def _loss_and_grads(
        self,
        state,
        frozen_state,
        rng,
        left_observation,
        right_observation,
        left_actions,
        right_actions,
        *,
        preprocessed,
        ownership_layout,
    ):
        model = nnx.merge(self.graphdef, nnx.State.merge(frozen_state, state.model_state))

        def loss_fn(current_model):
            result = current_model.compute_loss(
                rng,
                left_observation,
                right_observation,
                left_actions,
                right_actions,
                ownership=ownership_layout,
                mode=self.mode,
                train=True,
                preprocessed=preprocessed,
            )
            return result.total, (jnp.mean(result.left), jnp.mean(result.right))

        diff_state = nnx.DiffState(0, self.trainable_filter)
        (loss, (left_loss, right_loss)), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model)
        return loss, left_loss, right_loss, grads

    def _finish_step(self, state, grads, loss, left_loss, right_loss, local_finite, *, axis_name):
        local_grad_norm = optax.global_norm(grads)
        local_loss = loss
        local_left_loss = left_loss
        local_right_loss = right_loss
        if axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name)
            loss = jax.lax.pmean(loss, axis_name)
            left_loss = jax.lax.pmean(left_loss, axis_name)
            right_loss = jax.lax.pmean(right_loss, axis_name)
            finite = jax.lax.pmin(local_finite.astype(jnp.int32), axis_name).astype(bool)
        else:
            finite = local_finite
        params = state.model_state
        grad_norm = optax.global_norm(grads)
        updates, next_opt_state = self.tx.update(grads, state.opt_state, params)
        updates = jax.tree.map(lambda update: jnp.where(finite, update, jnp.zeros_like(update)), updates)
        next_opt_state = jax.tree.map(
            lambda new, old: jnp.where(finite, new, old),
            next_opt_state,
            state.opt_state,
        )
        next_params = optax.apply_updates(params, updates)
        next_state = EngineState(next_params, next_opt_state, state.step + finite.astype(jnp.int32))
        return next_state, StepInfo(
            loss,
            left_loss,
            right_loss,
            grad_norm,
            finite,
            local_loss,
            local_left_loss,
            local_right_loss,
            local_grad_norm,
        )

    def _step_impl(
        self,
        state,
        frozen_state,
        rng,
        left_observation,
        right_observation,
        left_actions,
        right_actions,
        *,
        preprocessed,
        ownership_layout,
        axis_name=None,
    ):
        loss, left_loss, right_loss, grads = self._loss_and_grads(
            state,
            frozen_state,
            rng,
            left_observation,
            right_observation,
            left_actions,
            right_actions,
            preprocessed=preprocessed,
            ownership_layout=ownership_layout,
        )
        local_finite = jnp.all(
            jnp.asarray(
                [
                    jnp.isfinite(loss),
                    jnp.isfinite(left_loss),
                    jnp.isfinite(right_loss),
                    jnp.isfinite(optax.global_norm(grads)),
                ]
            )
        )
        return self._finish_step(
            state,
            grads,
            loss,
            left_loss,
            right_loss,
            local_finite,
            axis_name=axis_name,
        )

    def _accumulated_step_impl(
        self,
        state,
        frozen_state,
        rng,
        left_observation,
        right_observation,
        left_actions,
        right_actions,
        *,
        preprocessed,
        ownership_layout,
        gradient_accumulation,
        axis_name=None,
    ):
        def split_microbatches(value):
            if value.shape[0] % gradient_accumulation:
                raise ValueError("local batch is not divisible by gradient_accumulation")
            micro_batch = value.shape[0] // gradient_accumulation
            return value.reshape((gradient_accumulation, micro_batch, *value.shape[1:]))

        left_observation = jax.tree.map(split_microbatches, left_observation)
        right_observation = jax.tree.map(split_microbatches, right_observation)
        left_actions = split_microbatches(left_actions)
        right_actions = split_microbatches(right_actions)
        ownership_layout = split_microbatches(ownership_layout)
        keys = jax.random.split(rng, gradient_accumulation)

        first_loss, first_left_loss, first_right_loss, first_grads = self._loss_and_grads(
            state,
            frozen_state,
            keys[0],
            jax.tree.map(lambda value: value[0], left_observation),
            jax.tree.map(lambda value: value[0], right_observation),
            left_actions[0],
            right_actions[0],
            preprocessed=preprocessed,
            ownership_layout=ownership_layout[0],
        )
        grad_sum = jax.tree.map(lambda grad: grad.astype(jnp.float32), first_grads)
        finite = jnp.all(
            jnp.asarray(
                [
                    jnp.isfinite(first_loss),
                    jnp.isfinite(first_left_loss),
                    jnp.isfinite(first_right_loss),
                    jnp.isfinite(optax.global_norm(first_grads)),
                ]
            )
        )

        def accumulate(carry, inputs):
            current_grad_sum, current_loss_sum, current_left_sum, current_right_sum, current_finite = carry
            key, left_obs, right_obs, left_act, right_act, current_ownership = inputs
            loss, left_loss, right_loss, grads = self._loss_and_grads(
                state,
                frozen_state,
                key,
                left_obs,
                right_obs,
                left_act,
                right_act,
                preprocessed=preprocessed,
                ownership_layout=current_ownership,
            )
            current_grad_sum = jax.tree.map(
                lambda total, grad: total + grad.astype(jnp.float32),
                current_grad_sum,
                grads,
            )
            current_finite = jnp.logical_and(
                current_finite,
                jnp.all(
                    jnp.asarray(
                        [
                            jnp.isfinite(loss),
                            jnp.isfinite(left_loss),
                            jnp.isfinite(right_loss),
                            jnp.isfinite(optax.global_norm(grads)),
                        ]
                    )
                ),
            )
            return (
                current_grad_sum,
                current_loss_sum + loss,
                current_left_sum + left_loss,
                current_right_sum + right_loss,
                current_finite,
            ), None

        tail_inputs = (
            keys[1:],
            jax.tree.map(lambda value: value[1:], left_observation),
            jax.tree.map(lambda value: value[1:], right_observation),
            left_actions[1:],
            right_actions[1:],
            ownership_layout[1:],
        )
        (grad_sum, loss_sum, left_sum, right_sum, finite), _ = jax.lax.scan(
            accumulate,
            (grad_sum, first_loss, first_left_loss, first_right_loss, finite),
            tail_inputs,
        )
        scale = jnp.asarray(gradient_accumulation, dtype=jnp.float32)
        grads = jax.tree.map(lambda grad: grad / scale, grad_sum)
        loss = loss_sum / scale
        left_loss = left_sum / scale
        right_loss = right_sum / scale
        return self._finish_step(
            state,
            grads,
            loss,
            left_loss,
            right_loss,
            finite,
            axis_name=axis_name,
        )

    def step(
        self,
        state,
        rng,
        left_observation,
        right_observation,
        left_actions,
        right_actions,
        ownership,
        *,
        preprocessed: bool = False,
        gradient_accumulation: int = 1,
    ):
        layout = np.asarray(ownership)
        if layout.ndim not in (1, 2):
            raise ValueError("ownership must have shape [tokens] or [batch, tokens]")
        ownership_layout = jnp.asarray(layout, dtype=jnp.int8)
        if gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        if self.distributed:
            key = (preprocessed, gradient_accumulation)
            if key not in self._parallel_steps:
                def parallel_step(state, frozen_state, device_rng, left_obs, right_obs, left_act, right_act, owner):
                    if gradient_accumulation == 1:
                        return self._step_impl(
                            state,
                            frozen_state,
                            device_rng,
                            left_obs,
                            right_obs,
                            left_act,
                            right_act,
                            preprocessed=preprocessed,
                            ownership_layout=owner,
                            axis_name="data",
                        )
                    return self._accumulated_step_impl(
                        state,
                        frozen_state,
                        device_rng,
                        left_obs,
                        right_obs,
                        left_act,
                        right_act,
                        preprocessed=preprocessed,
                        ownership_layout=owner,
                        gradient_accumulation=gradient_accumulation,
                        axis_name="data",
                    )

                pmap_kwargs = {"axis_name": "data"}
                if jax.process_count() == 1:
                    # Preserve deterministic placement for local multi-device tests.
                    pmap_kwargs["devices"] = self._distributed_devices
                    pmap_kwargs["in_axes"] = (0, None, 0, 0, 0, 0, 0, 0)
                else:
                    # Each process supplies one unreplicated model/optimizer
                    # state and one mapped local data shard.
                    pmap_kwargs["in_axes"] = (None, None, 0, 0, 0, 0, 0, 0)
                # Multi-process pmap must omit `devices` so the named axis spans
                # every process-local shard in the distributed world.
                self._parallel_steps[key] = jax.pmap(parallel_step, **pmap_kwargs)
            device_rngs = process_device_rngs(
                rng,
                process_index=jax.process_index(),
                process_count=jax.process_count(),
                local_device_count=self.device_count,
            )
            left_observation, right_observation, left_actions, right_actions, ownership_layout = shard_batch(
                (left_observation, right_observation, left_actions, right_actions, ownership_layout),
                device_count=self.device_count,
            )
            next_state, info = self._parallel_steps[key](
                state,
                self.frozen_state,
                device_rngs,
                left_observation,
                right_observation,
                left_actions,
                right_actions,
                ownership_layout,
            )
            if not self._state_has_replica_axis:
                next_state = unreplicate_tree(next_state)
            return next_state, info
        if gradient_accumulation == 1:
            return self._compiled_step(
                state,
                self.frozen_state,
                rng,
                left_observation,
                right_observation,
                left_actions,
                right_actions,
                preprocessed=preprocessed,
                ownership_layout=ownership_layout,
            )
        return self._compiled_accumulated_step(
            state,
            self.frozen_state,
            rng,
            left_observation,
            right_observation,
            left_actions,
            right_actions,
            preprocessed=preprocessed,
            ownership_layout=ownership_layout,
            gradient_accumulation=gradient_accumulation,
        )


def create_engine(model, *, mode: DualPi05Mode, tx: optax.GradientTransformation) -> TrainingEngine:
    return TrainingEngine(model, mode=mode, tx=tx)


def create_native_engine(
    model,
    *,
    mode: DualPi05Mode,
    settings,
    train_action_ffw: bool = True,
    train_action_attention: bool = True,
    train_paligemma_kv: bool = True,
) -> TrainingEngine:
    """Create a native-mode engine with disjoint strong parameter groups."""
    if not mode.is_native:
        raise ValueError("create_native_engine requires a native pi0.5 mode")
    trainable_filter = native_trainable_filter(
        train_action_ffw=train_action_ffw,
        train_action_attention=train_action_attention,
        train_paligemma_kv=train_paligemma_kv,
    )
    selected = nnx.state(model).filter(trainable_filter)
    labels = native_optimizer_labels(
        selected,
        train_action_ffw=train_action_ffw,
        train_action_attention=train_action_attention,
        train_paligemma_kv=train_paligemma_kv,
    )
    from pi05_fabric.training.optimizer import create_grouped_optimizer

    tx = create_grouped_optimizer(settings, labels)
    return TrainingEngine(
        model,
        mode=mode,
        tx=tx,
        trainable_filter=trainable_filter,
    )
