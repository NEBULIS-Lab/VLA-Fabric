import copy
import hashlib
import json

import pytest

from pi05_fabric.data.shared_mixture import SharedMixtureSpec
from pi05_fabric.data.shared_mixture import compare_matched_specs
from pi05_fabric.data.shared_mixture import task_counts_through_step
from pi05_fabric.data.shared_mixture import task_index_for_step


MANDATORY = ("aloha_handover_box", "aloha_shoes_table", "robotwin_handover_mic")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(tmp_path, name):
    root = tmp_path / name
    root.mkdir(parents=True)
    dataset_manifest = root / "manifest.json"
    dataset_manifest.write_text(json.dumps({
        "schema_version": 1,
        "episodes": [],
        "action_dim": 20,
        "state_dim": 20,
        "camera_keys": ["global", "left_wrist", "right_wrist"],
    }))
    dataset_sha = _sha(dataset_manifest)
    normalizer = root / "normalization_q01_q99.json"
    normalizer.write_text(json.dumps({
        "schema_version": 2,
        "normalization": "q01_q99",
        "dataset_manifest_sha256": dataset_sha,
    }))
    return {
        "name": name,
        "dataset": str(root),
        "dataset_manifest_sha256": dataset_sha,
        "normalizer": str(normalizer),
        "normalizer_sha256": _sha(normalizer),
        "evaluation_manifest": None,
        "evaluation_manifest_sha256": None,
        "action_dim": 20,
        "state_dim": 20,
        "camera_keys": ["global", "left_wrist", "right_wrist"],
    }


def _payload(tmp_path, *, count=3, condition="full"):
    names = list(MANDATORY)
    if count == 4:
        names.append("robotwin_scan_object")
    return {
        "schema_version": 1,
        "release_id": "shared-test",
        "condition": condition,
        "tasks": [_task(tmp_path, name) for name in names],
        "mixture_seed": 20260920,
        "model_seed": 20260824,
        "training_seed": 20260921,
        "optimizer": {"name": "grouped_adamw", "peak_learning_rate": 5e-5},
        "optimizer_steps": None,
        "checkpoint_steps": [],
        "formal_training_authorized": False,
        "writable_paths": {"output": f"outputs/{condition}", "logs": f"logs/{condition}"},
    }


def _write(tmp_path, payload, name="mixture.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("count", [3, 4])
def test_manifest_loads_three_or_four_audited_tasks(tmp_path, count):
    spec = SharedMixtureSpec.from_json(_write(tmp_path, _payload(tmp_path, count=count)))
    spec.validate_files()
    assert len(spec.tasks) == count
    assert tuple(task.name for task in spec.tasks[:3]) == MANDATORY
    assert spec.contract_dict()["tasks"][0]["name"] == MANDATORY[0]


def test_manifest_rejects_duplicate_task_names(tmp_path):
    payload = _payload(tmp_path)
    payload["tasks"][2]["name"] = payload["tasks"][1]["name"]
    with pytest.raises(ValueError, match="duplicate"):
        SharedMixtureSpec.from_json(_write(tmp_path, payload))


def test_manifest_rejects_reordered_mandatory_tasks(tmp_path):
    payload = _payload(tmp_path)
    payload["tasks"][0], payload["tasks"][1] = payload["tasks"][1], payload["tasks"][0]
    with pytest.raises(ValueError, match="mandatory task order"):
        SharedMixtureSpec.from_json(_write(tmp_path, payload))


def test_manifest_rejects_missing_files(tmp_path):
    payload = _payload(tmp_path)
    payload["tasks"][0]["dataset"] = str(tmp_path / "absent")
    spec = SharedMixtureSpec.from_json(_write(tmp_path, payload))
    with pytest.raises(FileNotFoundError, match="dataset manifest"):
        spec.validate_files()


def test_manifest_rejects_invalid_sha256(tmp_path):
    payload = _payload(tmp_path)
    payload["tasks"][0]["dataset_manifest_sha256"] = "short"
    with pytest.raises(ValueError, match="SHA-256"):
        SharedMixtureSpec.from_json(_write(tmp_path, payload))


@pytest.mark.parametrize("target", ["dataset", "normalizer"])
def test_manifest_rejects_changed_scientific_file(tmp_path, target):
    payload = _payload(tmp_path)
    spec = SharedMixtureSpec.from_json(_write(tmp_path, payload))
    task = payload["tasks"][0]
    path = (
        tmp_path / MANDATORY[0] / "manifest.json"
        if target == "dataset"
        else tmp_path / MANDATORY[0] / "normalization_q01_q99.json"
    )
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match=f"{target}.*hash"):
        spec.validate_files()


def test_manifest_rejects_normalizer_bound_to_another_dataset(tmp_path):
    payload = _payload(tmp_path)
    path = tmp_path / MANDATORY[0] / "normalization_q01_q99.json"
    normalizer = json.loads(path.read_text())
    normalizer["dataset_manifest_sha256"] = "f" * 64
    path.write_text(json.dumps(normalizer))
    payload["tasks"][0]["normalizer_sha256"] = _sha(path)
    spec = SharedMixtureSpec.from_json(_write(tmp_path, payload))
    with pytest.raises(ValueError, match="normalizer.*dataset"):
        spec.validate_files()


def test_manifest_hashes_optional_evaluation_manifest(tmp_path):
    payload = _payload(tmp_path)
    evaluation = tmp_path / "box-evaluation.json"
    evaluation.write_text(json.dumps({"split": "development", "condition_ids": [0, 1]}))
    payload["tasks"][0]["evaluation_manifest"] = str(evaluation)
    payload["tasks"][0]["evaluation_manifest_sha256"] = _sha(evaluation)
    spec = SharedMixtureSpec.from_json(_write(tmp_path, payload))
    spec.validate_files()

    evaluation.write_text(evaluation.read_text() + "\n")
    with pytest.raises(ValueError, match="evaluation manifest hash"):
        spec.validate_files()


def test_manifest_rejects_unhashed_evaluation_manifest(tmp_path):
    payload = _payload(tmp_path)
    evaluation = tmp_path / "box-evaluation.json"
    evaluation.write_text(json.dumps({"split": "development", "condition_ids": [0, 1]}))
    payload["tasks"][0]["evaluation_manifest"] = str(evaluation)
    with pytest.raises(ValueError, match="evaluation manifest and hash"):
        SharedMixtureSpec.from_json(_write(tmp_path, payload))


@pytest.mark.parametrize("field,value", [("action_dim", 18), ("state_dim", 18)])
def test_manifest_rejects_mixed_dimensions(tmp_path, field, value):
    payload = _payload(tmp_path)
    payload["tasks"][-1][field] = value
    with pytest.raises(ValueError, match=field):
        SharedMixtureSpec.from_json(_write(tmp_path, payload))


def test_authorized_manifest_requires_positive_budget(tmp_path):
    payload = _payload(tmp_path)
    payload["formal_training_authorized"] = True
    with pytest.raises(ValueError, match="optimizer_steps"):
        SharedMixtureSpec.from_json(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("checkpoint_steps", []),
        lambda payload: payload.__setitem__("checkpoint_steps", [50]),
        lambda payload: payload.__setitem__("optimizer", {}),
        lambda payload: payload["optimizer"].pop("batch_size"),
    ],
)
def test_full_only_authorization_requires_complete_schedule(tmp_path, mutation):
    payload = _payload(tmp_path, count=4)
    payload["formal_training_authorized"] = True
    payload["optimizer_steps"] = 100
    payload["checkpoint_steps"] = [50, 100]
    payload["optimizer"] = {
        "learning_rate": 5e-5,
        "final_learning_rate": 5e-6,
        "warmup_steps": 10,
        "batch_size": 4,
        "device_count": 4,
        "gradient_accumulation": 1,
    }
    mutation(payload)
    with pytest.raises(ValueError, match="authorized training"):
        SharedMixtureSpec.from_json(_write(tmp_path, payload))


def test_matched_conditions_allow_only_identity_and_writable_paths(tmp_path):
    full_payload = _payload(tmp_path / "full", condition="full")
    independent_payload = copy.deepcopy(full_payload)
    independent_payload["condition"] = "independent"
    independent_payload["release_id"] = "shared-test-independent"
    independent_payload["writable_paths"] = {
        "output": "outputs/independent", "logs": "logs/independent"
    }
    full = SharedMixtureSpec.from_json(_write(tmp_path, full_payload, "full.json"))
    independent = SharedMixtureSpec.from_json(
        _write(tmp_path, independent_payload, "independent.json")
    )
    compare_matched_specs(full, independent)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["tasks"].reverse(),
        lambda payload: payload.__setitem__("mixture_seed", 9),
        lambda payload: payload["optimizer"].__setitem__("peak_learning_rate", 1e-4),
        lambda payload: payload.__setitem__("optimizer_steps", 100),
        lambda payload: payload.__setitem__("checkpoint_steps", [50, 100]),
    ],
)
def test_matched_conditions_reject_scientific_difference(tmp_path, mutation):
    full_payload = _payload(tmp_path / "full", condition="full")
    independent_payload = copy.deepcopy(full_payload)
    independent_payload["condition"] = "independent"
    independent_payload["release_id"] = "other"
    independent_payload["writable_paths"] = {"output": "other", "logs": "other"}
    mutation(independent_payload)
    full = SharedMixtureSpec.from_json(_write(tmp_path, full_payload, "full.json"))
    try:
        independent = SharedMixtureSpec.from_json(
            _write(tmp_path, independent_payload, "independent.json")
        )
    except ValueError:
        return
    with pytest.raises(ValueError, match="matched"):
        compare_matched_specs(full, independent)


@pytest.mark.parametrize("task_count", [3, 4])
def test_scheduler_covers_every_task_once_per_block(task_count):
    for block in range(12):
        selected = [
            task_index_for_step(task_count, seed=20260920, step=block * task_count + offset)
            for offset in range(task_count)
        ]
        assert sorted(selected) == list(range(task_count))


def test_scheduler_is_stateless_and_rank_independent():
    expected = [task_index_for_step(4, seed=17, step=step) for step in range(40)]
    for _rank in range(8):
        replay = [task_index_for_step(4, seed=17, step=step) for step in range(40)]
        assert replay == expected


def test_scheduler_seed_changes_sequence():
    first = [task_index_for_step(4, seed=1, step=step) for step in range(40)]
    second = [task_index_for_step(4, seed=2, step=step) for step in range(40)]
    assert first != second


@pytest.mark.parametrize("task_count", [0, -1])
def test_scheduler_rejects_nonpositive_task_count(task_count):
    with pytest.raises(ValueError, match="task_count"):
        task_index_for_step(task_count, seed=1, step=0)


def test_scheduler_rejects_negative_step():
    with pytest.raises(ValueError, match="step"):
        task_index_for_step(3, seed=1, step=-1)


def test_task_counts_support_non_block_boundary_resume():
    names = ("box", "shoes", "mic", "scan")
    completed_steps = 11
    counts = task_counts_through_step(names, seed=20260920, completed_steps=completed_steps)
    expected = {name: 0 for name in names}
    for step in range(completed_steps):
        expected[names[task_index_for_step(4, seed=20260920, step=step)]] += 1
    assert counts == expected
    assert sorted(counts.values()) == [2, 3, 3, 3]


def test_task_counts_reject_duplicate_names_or_negative_steps():
    with pytest.raises(ValueError, match="unique"):
        task_counts_through_step(("box", "box"), seed=1, completed_steps=1)
    with pytest.raises(ValueError, match="completed_steps"):
        task_counts_through_step(("box", "shoes"), seed=1, completed_steps=-1)
