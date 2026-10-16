"""Portable frame-task contracts, without internal run-directory dependencies."""

import numpy as np
import pytest

from pi05_fabric.data.multiarm_frame_tasks import MultiArmQuantileNormalization
from pi05_fabric.training.stages import StageName


@pytest.mark.parametrize("count", [3, 4])
def test_role_normalization_roundtrip(tmp_path, count):
    path = tmp_path / "quantiles.npz"
    arrays = {}
    for role in range(count):
        for field, width in [("state", 9), ("action", 7)]:
            arrays[f"role_{role}_{field}_q01"] = -np.ones(width)
            arrays[f"role_{role}_{field}_q99"] = np.ones(width)
    np.savez(path, **arrays)
    norm = MultiArmQuantileNormalization.from_npz(path, expected_agent_count=count)
    assert norm.agent_count == count
    np.testing.assert_allclose(norm.unnormalize_actions(0, np.zeros((2, 7))), 0, atol=1e-6)
    with pytest.raises(ValueError):
        norm.for_role(count)


def test_bimanual_and_frame_stages_coexist():
    stages = {stage.value for stage in StageName}
    assert {"pi_native_v2_independent", "pi_native_v2_full_direct",
            "pi05_frame3_full_stage2", "pi05_frame3_independent_direct",
            "pi05_frame4_independent_direct", "pi05_four_arm_full_stage2"} <= stages
