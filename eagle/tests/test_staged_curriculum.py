from types import SimpleNamespace

import pytest

from commvla.training.train_dual_arm import (
    curriculum_phase_for_step,
    expected_communication_groups,
    validate_communication_groups,
)


def staged_args(**overrides):
    values = {
        "interaction_curriculum": "common_only_then_stage2",
        "curriculum_transition_step": 10000,
        "common_kv_mode": "local",
        "remote_kv_mode": "full",
        "action_token_fusion_mode": "none",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_curriculum_boundary_uses_one_continuous_global_step_axis() -> None:
    args = staged_args()
    assert curriculum_phase_for_step(args, 1) == "stage1_common_only"
    assert curriculum_phase_for_step(args, 10000) == "stage1_common_only"
    assert curriculum_phase_for_step(args, 10001) == "stage2"
    assert curriculum_phase_for_step(args, 40000) == "stage2"


def test_legacy_continuous_stage_expected_communication_profiles() -> None:
    args = staged_args()
    assert expected_communication_groups(args, "stage1_common_only") == {
        "common": True,
        "private_kv": False,
        "action_intent": False,
    }
    assert expected_communication_groups(args, "stage2") == {
        "common": False,
        "private_kv": True,
        "action_intent": False,
    }


def test_staged_communication_gate_rejects_leaked_channel() -> None:
    expected = {"common": True, "private_kv": False, "action_intent": False}
    validate_communication_groups(
        {"common": 1024, "private_kv": 0, "action_intent": 0, "total": 1024},
        expected,
        label="stage1_common_only",
    )
    with pytest.raises(RuntimeError, match="Unexpected stage1_common_only"):
        validate_communication_groups(
            {"common": 1024, "private_kv": 256, "action_intent": 0, "total": 1280},
            expected,
            label="stage1_common_only",
        )
