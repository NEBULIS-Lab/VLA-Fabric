from flax import nnx
import jax
import jax.numpy as jnp
from types import SimpleNamespace

from pi05_fabric.agents.multiagent_pi05 import LocalRolePi05
from pi05_fabric.agents.multiagent_pi05 import MultiAgentPi05Mode
from pi05_fabric.agents.multiagent_pi05 import MultiAgentResidualActionBranch
from pi05_fabric.agents.multiagent_pi05 import fold_role_rng
from pi05_fabric.agents.multiagent_pi05 import team_loss_reference


def test_full_mode_enables_all_three_paths_and_common_only_does_not():
    assert MultiAgentPi05Mode.FULL.common
    assert MultiAgentPi05Mode.FULL.private_kv
    assert MultiAgentPi05Mode.FULL.residual_action
    assert MultiAgentPi05Mode.COMMON_ONLY.common
    assert not MultiAgentPi05Mode.COMMON_ONLY.private_kv
    assert not MultiAgentPi05Mode.COMMON_ONLY.residual_action


def test_residual_branch_has_three_role_ordered_peer_slots():
    branch = MultiAgentResidualActionBranch(
        depth=18,
        width=8,
        peer_width=4,
        max_agents=4,
        rngs=nnx.Rngs(0),
    )
    state = nnx.state(branch).to_pure_dict()

    assert state["kernel"].shape == (18, 12, 8)
    assert state["gate"].shape == (18,)
    assert jnp.array_equal(state["gate"], jnp.zeros((18,)))


def test_team_loss_uses_mean_without_hiding_role_losses():
    losses = jnp.asarray([1.0, 2.0, 6.0])

    total, roles = team_loss_reference(losses)

    assert float(total) == 3.0
    assert roles.tolist() == [1.0, 2.0, 6.0]


def test_role_rngs_are_independent_while_team_time_key_is_shared():
    key = jax.random.key(7)
    role0 = fold_role_rng(key, role_index=0)
    role1 = fold_role_rng(key, role_index=1)

    assert jnp.array_equal(role0.time, role1.time)
    assert not jnp.array_equal(role0.noise, role1.noise)
    assert not jnp.array_equal(role0.preprocess, role1.preprocess)


def test_local_role_wrapper_owns_exactly_one_complete_policy():
    config = SimpleNamespace(
        depth=18,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    policy = SimpleNamespace(
        action_in_proj=SimpleNamespace(out_features=8),
        PaliGemma=SimpleNamespace(
            llm=SimpleNamespace(module=SimpleNamespace(configs=(config, config)))
        ),
    )

    wrapper = LocalRolePi05(
        policy,
        agent_count=3,
        max_agents=4,
        rngs=nnx.Rngs(0),
    )

    assert wrapper.policy is policy
    assert not hasattr(wrapper, "peer_policy")
    state = nnx.state(wrapper.residual_action).to_pure_dict()
    assert state["kernel"].shape == (18, 24, 8)
    assert state["gate"].shape == (18,)
