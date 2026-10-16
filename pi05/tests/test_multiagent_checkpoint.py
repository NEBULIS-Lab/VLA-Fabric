from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from pi05_fabric.training.multiagent_checkpoint import audit_team_checkpoint
from pi05_fabric.training.multiagent_checkpoint import finalize_team_checkpoint
from pi05_fabric.training.multiagent_checkpoint import load_role_weights
from pi05_fabric.training.multiagent_checkpoint import save_role_checkpoint
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import TrainingSnapshot


def snapshot(role: int):
    return TrainingSnapshot(
        stage=StageName.PI05_STACKCUBE_3A_COMMON_STAGE1,
        step=10,
        schedule_step=10,
        params={"role_weight": jnp.asarray([role, role + 1], dtype=jnp.float32)},
        opt_state=(jnp.asarray(role, dtype=jnp.int32),),
        rng=jax.random.key(100 + role),
        parent_model_sha256=None,
        model_seed=7 + role,
        training_seed=19,
        protocol_metadata={"agent_count": 3, "role": role},
    )


def test_team_checkpoint_requires_every_role_before_complete(tmp_path):
    root = tmp_path / "step-10"
    save_role_checkpoint(root, role=0, snapshot=snapshot(0))
    save_role_checkpoint(root, role=1, snapshot=snapshot(1))

    with pytest.raises(ValueError, match="missing role shards"):
        finalize_team_checkpoint(root, agent_count=3, protocol_hash="abc")

    assert not (root / "COMPLETE").exists()


def test_complete_team_checkpoint_restores_only_requested_role_weights(tmp_path):
    root = tmp_path / "step-10"
    for role in range(3):
        save_role_checkpoint(root, role=role, snapshot=snapshot(role))
    finalize_team_checkpoint(root, agent_count=3, protocol_hash="abc")

    audit = audit_team_checkpoint(root, expected_agent_count=3)
    role2 = load_role_weights(root, role=2, expected_agent_count=3)

    assert audit["complete"] is True
    assert audit["step"] == 10
    assert audit["roles"] == [0, 1, 2]
    assert role2["role_weight"].tolist() == [2.0, 3.0]


def test_role_checkpoint_cannot_be_overwritten(tmp_path):
    root = tmp_path / "step-10"
    save_role_checkpoint(root, role=0, snapshot=snapshot(0))

    with pytest.raises(FileExistsError):
        save_role_checkpoint(root, role=0, snapshot=snapshot(0))
