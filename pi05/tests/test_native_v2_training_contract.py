from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.training.launch_config import LaunchConfig
from pi05_fabric.training.stages import StageName, TrainingSnapshot, fork_stage2


def _snapshot(stage, *, training_seed=11):
    return TrainingSnapshot(
        stage=stage,
        step=10,
        schedule_step=10,
        params={"weight": jnp.ones((2,))},
        opt_state={},
        rng=jax.random.key(training_seed),
        parent_model_sha256=None,
        model_seed=7,
        training_seed=training_seed,
    )


def test_v2_stages_map_to_v2_modes(tmp_path: Path):
    independent = LaunchConfig(
        stage=StageName.PI_NATIVE_V2_INDEPENDENT,
        dataset=tmp_path,
        base_checkpoint=tmp_path,
        output=tmp_path / "independent",
        steps=50,
        batch_size=1,
        learning_rate=5e-5,
        warmup_steps=2,
        model_seed=7,
        training_seed=10,
    )
    assert independent.mode is DualPi05Mode.PI_NATIVE_V2_INDEPENDENT
    assert independent.mode.is_v2
    assert independent.mode.interaction_spec == (False, False, False, False)
    independent.validate_paths(require_existing=False)

    stage1 = LaunchConfig(
        stage=StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1,
        dataset=tmp_path,
        base_checkpoint=tmp_path,
        output=tmp_path / "stage1",
        steps=10,
        batch_size=1,
        learning_rate=5e-5,
        warmup_steps=1,
        model_seed=7,
        training_seed=11,
    )
    assert stage1.mode is DualPi05Mode.PI_NATIVE_V2_RAW_COMMON_ONLY

    stage2 = LaunchConfig(
        stage=StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        dataset=tmp_path,
        base_checkpoint=tmp_path,
        output=tmp_path / "stage2",
        steps=40,
        batch_size=1,
        learning_rate=5e-5,
        warmup_steps=2,
        model_seed=7,
        training_seed=12,
        parent_training_seed=11,
        stage1_checkpoint=tmp_path / "stage1-ckpt",
    )
    assert stage2.mode is DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION
    stage2.validate_paths(require_existing=False)


def test_v2_stage2_requires_v2_stage1_and_distinct_rng():
    stage1 = _snapshot(StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1)
    stage2 = fork_stage2(
        stage1,
        target=StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        optimizer_init=lambda params: {"state": params},
        rng=jax.random.key(12),
        training_seed=12,
    )
    assert stage2.step == 0
    assert stage2.training_seed == 12

    with pytest.raises(ValueError, match="must differ"):
        fork_stage2(
            stage1,
            target=StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
            optimizer_init=lambda params: {},
            rng=jax.random.key(11),
            training_seed=11,
        )
    with pytest.raises(ValueError, match="incompatible"):
        fork_stage2(
            _snapshot(StageName.PI_NATIVE_RAW_COMMON_STAGE1),
            target=StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
            optimizer_init=lambda params: {},
            rng=jax.random.key(12),
            training_seed=12,
        )
