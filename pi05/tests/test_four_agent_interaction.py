from types import SimpleNamespace

from flax import nnx
import jax.numpy as jnp

from pi05_fabric.agents.multiagent_pi05 import LocalRolePi05
from pi05_fabric.communication.multiagent_collectives import AgentGroupSpec
from pi05_fabric.communication.multiagent_collectives import pack_peer_contexts
from pi05_fabric.communication.multiagent_collectives import reference_fuse_common
from pi05_fabric.communication.multiagent_gemma import MultiAgentGemmaTrace
from pi05_fabric.communication.multiagent_gemma import residual_peer_contexts


def test_four_agent_group_uses_one_role_per_rank() -> None:
    spec = AgentGroupSpec.from_world_size(world_size=4, agent_count=4)

    assert spec.agent_groups == ((0, 1, 2, 3),)
    assert spec.role_data_parallel_groups == ((0,), (1,), (2,), (3,))


def test_four_agent_common_fuses_only_common_owned_positions() -> None:
    values = jnp.asarray(
        [
            [[[1.0], [10.0]]],
            [[[3.0], [20.0]]],
            [[[5.0], [30.0]]],
            [[[7.0], [40.0]]],
        ]
    )
    fused = reference_fuse_common(values, jnp.asarray([0, 1]), common_value=0)

    assert fused[:, 0, 0, 0].tolist() == [4.0, 4.0, 4.0, 4.0]
    assert fused[:, 0, 1, 0].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_four_agent_action_contexts_preserve_three_sender_slots() -> None:
    contexts = jnp.asarray([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])

    packed, valid = pack_peer_contexts(contexts, receiver_role=2, max_agents=4)

    assert packed.shape == (1, 1, 3)
    assert packed[0, 0].tolist() == [1.0, 2.0, 4.0]
    assert valid.tolist() == [True, True, True]


def test_three_action_peers_use_independent_softmax_without_averaging() -> None:
    query = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
    peer_keys = jnp.ones((3, 1, 1, 1, 1), dtype=jnp.float32)
    peer_values = jnp.asarray([2.0, 5.0, 11.0], dtype=jnp.float32).reshape(3, 1, 1, 1, 1)
    peer_valid = jnp.ones((3, 1, 1), dtype=jnp.bool_)

    contexts = residual_peer_contexts(
        query,
        peer_keys,
        peer_values,
        peer_valid,
        num_kv_heads=1,
    )

    assert contexts.shape == (1, 1, 3)
    assert contexts[0, 0].tolist() == [2.0, 5.0, 11.0]


def test_four_agent_local_wrapper_owns_one_policy_and_three_peer_projection_slots() -> None:
    config = SimpleNamespace(depth=18, num_heads=2, num_kv_heads=1, head_dim=4)
    policy = SimpleNamespace(
        action_in_proj=SimpleNamespace(out_features=8),
        PaliGemma=SimpleNamespace(
            llm=SimpleNamespace(module=SimpleNamespace(configs=(config, config)))
        ),
    )

    wrapper = LocalRolePi05(
        policy,
        agent_count=4,
        max_agents=4,
        rngs=nnx.Rngs(0),
    )
    state = nnx.state(wrapper.residual_action).to_pure_dict()

    assert wrapper.policy is policy
    assert not hasattr(wrapper, "peer_policy")
    assert state["kernel"].shape == (18, 24, 8)


def test_trace_explicitly_reports_no_query_or_raw_observation_transport() -> None:
    trace = MultiAgentGemmaTrace(
        raw_event_counts=jnp.ones((18, 7), dtype=jnp.int32),
        remote_private_tokens=jnp.asarray(12),
        residual_action_tokens=60,
        peer_slots=3,
    )

    assert trace.transmitted_queries == 0
    assert trace.transmitted_raw_observations == 0
