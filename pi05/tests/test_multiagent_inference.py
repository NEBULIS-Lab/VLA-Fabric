from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0_config

from pi05_fabric.agents.multiagent_pi05 import LocalRolePi05
from pi05_fabric.agents.multiagent_pi05 import MultiAgentPi05Mode
from pi05_fabric.agents.pi05_strong import native_trainable_filter
from pi05_fabric.evaluation.multiagent_inference import MultiAgentInferenceEngine
from pi05_fabric.training.agent_parallel import AgentParallelTopology


def _model():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    return LocalRolePi05(
        config.create(jax.random.key(10)),
        agent_count=3,
        max_agents=4,
        rngs=nnx.Rngs(20),
    )


def test_inference_engine_restores_only_checkpoint_trainable_state():
    model = _model()
    trainable_filter = native_trainable_filter(
        train_action_ffw=False,
        train_action_attention=True,
        train_paligemma_kv=True,
    )
    selected = nnx.state(model).filter(trainable_filter).to_pure_dict()
    selected["residual_action"]["gate"] = jnp.ones_like(
        selected["residual_action"]["gate"]
    )

    engine = MultiAgentInferenceEngine(
        model,
        checkpoint_params=selected,
        mode=MultiAgentPi05Mode.FULL,
        topology=AgentParallelTopology.create(world_size=3, agent_count=3),
    )

    restored = engine.selected_params().to_pure_dict()
    assert bool(jnp.all(restored["residual_action"]["gate"] == 1))


def test_inference_engine_requires_three_processes_before_sampling():
    model = _model()
    trainable_filter = native_trainable_filter(
        train_action_ffw=False,
        train_action_attention=True,
        train_paligemma_kv=True,
    )
    engine = MultiAgentInferenceEngine(
        model,
        checkpoint_params=nnx.state(model).filter(trainable_filter).to_pure_dict(),
        mode=MultiAgentPi05Mode.FULL,
        topology=AgentParallelTopology.create(world_size=3, agent_count=3),
    )

    try:
        engine.validate_runtime()
    except ValueError as error:
        assert "process count" in str(error)
    else:
        raise AssertionError("single-process tests must not pass the three-rank gate")
