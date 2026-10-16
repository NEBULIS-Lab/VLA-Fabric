import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from openpi.models import pi0_config

from pi05_fabric.agents.multiagent_pi05 import LocalRolePi05
from pi05_fabric.agents.multiagent_pi05 import MultiAgentPi05Mode
from pi05_fabric.training.agent_parallel import AgentParallelTopology
from pi05_fabric.training.agent_parallel import batch_configuration
from pi05_fabric.training.agent_parallel import team_rng
from pi05_fabric.training.multiagent_engine import create_multiagent_engine
from pi05_fabric.training.multiagent_engine import finite_gate_reference
from pi05_fabric.training.optimizer import GroupedOptimizerSettings


def test_three_rank_topology_has_one_agent_group_and_role_local_dp_groups():
    topology = AgentParallelTopology.create(world_size=3, agent_count=3)

    assert topology.agent_groups == ((0, 1, 2),)
    assert topology.role_data_parallel_groups == ((0,), (1,), (2,))
    assert [topology.role_for_rank(rank) for rank in range(3)] == [0, 1, 2]
    assert [topology.team_for_rank(rank) for rank in range(3)] == [0, 0, 0]


def test_six_rank_topology_creates_two_aligned_teams():
    topology = AgentParallelTopology.create(world_size=6, agent_count=3)

    assert topology.agent_groups == ((0, 1, 2), (3, 4, 5))
    assert topology.role_data_parallel_groups == ((0, 3), (1, 4), (2, 5))
    assert topology.role_for_rank(5) == 2
    assert topology.team_for_rank(5) == 1


@pytest.mark.parametrize(
    ("microbatch", "accumulation"),
    [(1, 12), (2, 6), (3, 4), (4, 3)],
)
def test_batch_candidates_all_preserve_global_team_batch(microbatch, accumulation):
    config = batch_configuration(
        team_microbatch=microbatch,
        accumulation=accumulation,
        data_parallel_teams=1,
    )

    assert config.global_team_batch == 12


def test_team_rng_is_shared_by_roles_but_differs_between_teams():
    key0 = team_rng(seed=17, step=4, microstep=2, team_index=0)
    key0_again = team_rng(seed=17, step=4, microstep=2, team_index=0)
    key1 = team_rng(seed=17, step=4, microstep=2, team_index=1)

    assert jnp.array_equal(key0, key0_again)
    assert not jnp.array_equal(key0, key1)


def test_incomplete_agent_group_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        AgentParallelTopology.create(world_size=4, agent_count=3)


def test_finite_gate_rejects_one_failed_role():
    assert finite_gate_reference([True, True, True])
    assert not finite_gate_reference([True, False, True])


def test_engine_uses_interaction_focused_boundary_and_trains_residual_branch():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = LocalRolePi05(
        config.create(jax.random.key(1)),
        agent_count=3,
        max_agents=4,
        rngs=nnx.Rngs(2),
    )
    engine = create_multiagent_engine(
        model,
        mode=MultiAgentPi05Mode.FULL,
        topology=AgentParallelTopology.create(world_size=3, agent_count=3),
        settings=GroupedOptimizerSettings(total_steps=10, warmup_steps=2),
    )
    paths = {
        "/".join(str(part) for part in path)
        for path, _ in engine.selected_params().flat_state().items()
    }

    assert any("residual_action/kernel" in path for path in paths)
    assert any("q_einsum_1/w" in path for path in paths)
    assert any("kv_einsum/w" in path and "kv_einsum_1" not in path for path in paths)
    assert not any("mlp_1/gating_einsum" in path and "lora" not in path for path in paths)
