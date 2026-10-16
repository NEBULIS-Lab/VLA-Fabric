"""Three-rank inference engine for role-sharded PI0.5 checkpoints."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from pi05_fabric.agents.multiagent_pi05 import MultiAgentPi05Mode
from pi05_fabric.agents.pi05_strong import native_trainable_filter
from pi05_fabric.training.agent_parallel import AgentParallelTopology
from pi05_fabric.training.distributed import shard_batch


class MultiAgentInferenceEngine:
    def __init__(
        self,
        model,
        *,
        checkpoint_params,
        mode: MultiAgentPi05Mode,
        topology: AgentParallelTopology,
    ):
        self.mode = mode
        self.topology = topology
        self.trainable_filter = native_trainable_filter(
            train_action_ffw=False,
            train_action_attention=True,
            train_paligemma_kv=True,
        )
        self.graphdef, state = nnx.split(model)
        selected = state.filter(self.trainable_filter)
        selected.replace_by_pure_dict(checkpoint_params)
        frozen = state.filter(nnx.Not(self.trainable_filter))
        self.model_state = nnx.State.merge(frozen, selected)
        self._parallel_sample = None

    def selected_params(self):
        return self.model_state.filter(self.trainable_filter)

    def validate_runtime(self) -> None:
        if jax.process_count() != self.topology.world_size:
            raise ValueError("JAX process count does not match the evaluation topology")
        if jax.local_device_count() != 1:
            raise ValueError("each evaluation process must expose exactly one GPU")

    def sample_actions(
        self,
        rng,
        observation,
        ownership,
        *,
        num_steps: int = 10,
        preprocessed: bool = False,
    ):
        self.validate_runtime()
        if self._parallel_sample is None:

            def sample(state, local_rng, local_observation, local_ownership):
                model = nnx.merge(self.graphdef, state)
                return model.sample_actions(
                    local_rng,
                    local_observation,
                    ownership=local_ownership,
                    mode=self.mode,
                    axis_name="agents",
                    agent_groups=self.topology.agent_groups,
                    num_steps=num_steps,
                    preprocessed=preprocessed,
                )

            self._parallel_sample = jax.pmap(
                sample,
                axis_name="agents",
                in_axes=(None, 0, 0, 0),
            )
        sharded_observation, sharded_ownership = shard_batch(
            (observation, ownership),
            device_count=1,
        )
        actions = self._parallel_sample(
            self.model_state,
            jnp.expand_dims(rng, axis=0),
            sharded_observation,
            sharded_ownership,
        )
        return actions[0]
