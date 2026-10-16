import csv
import json

import numpy as np
import pytest

from pi05_fabric.evaluation.shared_four_task import SHARED_TASKS
from pi05_fabric.evaluation.shared_four_task import SharedTaskAdapter
from pi05_fabric.evaluation.shared_four_task import evaluate_shared_task


def _conditions(count=200):
    return {
        "schema_version": 1,
        "task": SHARED_TASKS[0],
        "split": "fresh",
        "count": count,
        "environment_protocol_id": "tabletop-box-v1",
        "conditions": [
            {"condition_id": condition_id, "target": condition_id % 2}
            for condition_id in range(count)
        ],
    }


def _adapter(*, error_at=None, timeout_at=None, nonfinite_at=None):
    def load_condition(condition):
        return {"condition": condition, "steps": 0, "success": False}

    def make_observation(state):
        return {"value": state["steps"]}

    def step(state, action):
        condition_id = state["condition"]["condition_id"]
        if condition_id == error_at:
            raise RuntimeError("adapter failure")
        if condition_id == timeout_at:
            raise TimeoutError("deadline")
        state = dict(state)
        state["steps"] += 1
        state["success"] = bool(np.asarray(action).sum() >= 0)
        return state, True

    return SharedTaskAdapter(
        name=SHARED_TASKS[0],
        environment_protocol="tabletop-box-v1",
        load_condition=load_condition,
        make_observation=make_observation,
        step=step,
        success=lambda state: state["success"],
    ), nonfinite_at


def _policy(nonfinite_at=None):
    def predict(_observation, *, condition_id, planning_call):
        del planning_call
        if condition_id == nonfinite_at:
            return np.asarray([np.nan])
        return np.asarray([1.0])

    return predict


def test_registry_contains_exactly_the_frozen_four_tasks() -> None:
    assert SHARED_TASKS == (
        "aloha_handover_box",
        "aloha_shoes_table",
        "robotwin_handover_mic",
        "robotwin_scan_object",
    )


def test_development_summary_is_compatible_with_checkpoint_selector(tmp_path) -> None:
    from pi05_fabric.evaluation.shared_four_task import evaluate_shared_task
    adapter, _ = _adapter()
    conditions = _conditions()
    conditions["split"] = "development"
    result = evaluate_shared_task(
        architecture="pi05", adapter=adapter, policy=_policy(),
        condition_manifest=conditions, mode="full", checkpoint_sha256="a" * 64,
        effective_channels={"common": True, "private_kv": True, "action": True},
        output_dir=tmp_path / "development", formal=True, checkpoint_step=10000,
    )
    assert result["checkpoint_step"] == 10000
    assert result["attempts"] == 200
    assert result["condition_ids"] == list(range(200))


def test_formal_evaluation_writes_normalized_rows_and_summary(tmp_path) -> None:
    adapter, _ = _adapter()

    result = evaluate_shared_task(
        architecture="pi05",
        adapter=adapter,
        policy=_policy(),
        condition_manifest=_conditions(),
        mode="all_off",
        checkpoint_sha256="a" * 64,
        effective_channels={"common": False, "private_kv": False, "action": False},
        output_dir=tmp_path / "result",
        formal=True,
    )

    with (tmp_path / "result/rollouts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 200
    assert set(rows[0]) >= {
        "architecture", "task", "mode", "condition_id", "success", "steps",
        "planning_calls", "timeout", "finite", "error", "checkpoint_sha256",
    }
    assert result["successes"] == 200
    assert result["expected_condition_ids"] == list(range(200))
    manifest = json.loads((tmp_path / "result/evaluation_manifest.json").read_text())
    assert manifest["effective_channels"] == {
        "common": False, "private_kv": False, "action": False
    }


def test_smoke_accepts_one_condition_and_preserves_failures(tmp_path) -> None:
    adapter, _ = _adapter(error_at=0)
    manifest = _conditions(1)

    result = evaluate_shared_task(
        architecture="pi05",
        adapter=adapter,
        policy=_policy(),
        condition_manifest=manifest,
        mode="full",
        checkpoint_sha256="b" * 64,
        effective_channels={"common": True, "private_kv": True, "action": True},
        output_dir=tmp_path / "error",
        formal=False,
    )

    assert result["errors"] == 1
    row = list(csv.DictReader((tmp_path / "error/rollouts.csv").open()))[0]
    assert row["success"] == "0"
    assert row["error"] == "adapter failure"


@pytest.mark.parametrize("failure", ["timeout", "nonfinite"])
def test_preserves_timeout_and_nonfinite_outcomes(tmp_path, failure) -> None:
    adapter, _ = _adapter(timeout_at=0 if failure == "timeout" else None)
    result = evaluate_shared_task(
        architecture="pi05",
        adapter=adapter,
        policy=_policy(nonfinite_at=0 if failure == "nonfinite" else None),
        condition_manifest=_conditions(1),
        mode="full",
        checkpoint_sha256="c" * 64,
        effective_channels={"common": True, "private_kv": True, "action": True},
        output_dir=tmp_path / failure,
        formal=False,
    )
    assert result["timeouts"] == int(failure == "timeout")
    assert result["nonfinite"] == int(failure == "nonfinite")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["conditions"].pop(), "exactly 200"),
        (lambda payload: payload["conditions"].__setitem__(1, payload["conditions"][0]), "duplicate"),
        (lambda payload: payload.__setitem__("task", SHARED_TASKS[1]), "task"),
        (lambda payload: payload.__setitem__("environment_protocol_id", "wrong"), "protocol"),
    ],
)
def test_formal_evaluation_rejects_invalid_frozen_manifest(tmp_path, mutation, message) -> None:
    adapter, _ = _adapter()
    manifest = _conditions()
    mutation(manifest)
    with pytest.raises(ValueError, match=message):
        evaluate_shared_task(
            architecture="pi05",
            adapter=adapter,
            policy=_policy(),
            condition_manifest=manifest,
            mode="full",
            checkpoint_sha256="d" * 64,
            effective_channels={"common": True, "private_kv": True, "action": True},
            output_dir=tmp_path / "invalid",
            formal=True,
        )
