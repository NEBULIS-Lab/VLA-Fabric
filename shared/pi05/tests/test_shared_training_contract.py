from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.data.shared_mixture import compare_matched_specs
from pi05_fabric.data.shared_mixture import SharedMixtureSpec
from pi05_fabric.training.launch_config import LaunchConfig
from pi05_fabric.training.stages import StageName


TRAINER = Path(__file__).parents[1] / "scripts/train_dual_pi05.py"
LAUNCHER = Path(__file__).parents[1] / "scripts/launch_distributed_pi05.py"
MANDATORY = ("aloha_handover_box", "aloha_shoes_table", "robotwin_handover_mic")


def _config(tmp_path: Path, stage: StageName) -> LaunchConfig:
    return LaunchConfig(
        stage=stage,
        dataset=tmp_path,
        base_checkpoint=tmp_path,
        output=tmp_path / stage.value,
        steps=400,
        batch_size=1,
        device_count=1,
        gradient_accumulation=1,
        learning_rate=5e-5,
        final_learning_rate=5e-6,
        warmup_steps=10,
        checkpoint_every=100,
        model_seed=20260824,
        training_seed=20260921,
    )


def test_shared_full_is_direct_v2_three_path_mode(tmp_path):
    config = _config(tmp_path, StageName.PI_NATIVE_V2_SHARED_FULL)
    config.validate_paths()
    assert config.mode is DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION
    assert config.mode.is_native and config.mode.is_v2
    assert tuple(config.mode.interaction_spec) == (True, True, False, False)
    assert config.mode.remote_action_residual
    assert config.stage1_checkpoint is None


def test_shared_independent_is_direct_v2_without_interaction(tmp_path):
    config = _config(tmp_path, StageName.PI_NATIVE_V2_SHARED_INDEPENDENT)
    config.validate_paths()
    assert config.mode is DualPi05Mode.PI_NATIVE_V2_INDEPENDENT
    assert config.mode.is_native and config.mode.is_v2
    assert tuple(config.mode.interaction_spec) == (False, False, False, False)
    assert not config.mode.remote_action_residual
    assert config.stage1_checkpoint is None


def test_shared_full_allows_eight_gpu_global_batch_32_contract(tmp_path):
    config = replace(
        _config(tmp_path, StageName.PI_NATIVE_V2_SHARED_FULL),
        batch_size=2,
        device_count=8,
        gradient_accumulation=2,
    )
    config.validate_paths()
    assert (
        config.batch_size * config.device_count * config.gradient_accumulation
        == 32
    )


def test_non_shared_stage_rejects_accumulation_two(tmp_path):
    config = replace(
        _config(tmp_path, StageName.INDEPENDENT),
        gradient_accumulation=2,
    )
    with pytest.raises(ValueError, match="not authorized for this training stage"):
        config.validate_paths()


def test_shared_modes_use_v2_action_horizon_contract(tmp_path):
    for stage in (
        StageName.PI_NATIVE_V2_SHARED_FULL,
        StageName.PI_NATIVE_V2_SHARED_INDEPENDENT,
    ):
        config = _config(tmp_path, stage)
        action_horizon = 50 if config.mode.is_v2 else 20
        assert action_horizon == 50


@pytest.fixture
def trainer():
    spec = importlib.util.spec_from_file_location("shared_pi05_trainer", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mixture(tmp_path, *, count, condition):
    names = ["aloha_handover_box", "aloha_shoes_table", "robotwin_handover_mic"]
    if count == 4:
        names.append("robotwin_scan_object")
    tasks = []
    for name in names:
        root = tmp_path / name
        root.mkdir(parents=True)
        dataset_manifest = root / "manifest.json"
        dataset_manifest.write_text(json.dumps({"schema_version": 1, "episodes": []}))
        dataset_sha = _sha(dataset_manifest)
        normalizer = root / "normalization_q01_q99.json"
        normalizer.write_text(json.dumps({
            "schema_version": 2,
            "normalization": "q01_q99",
            "dataset_manifest_sha256": dataset_sha,
        }))
        tasks.append({
            "name": name,
            "dataset": str(root),
            "dataset_manifest_sha256": dataset_sha,
            "normalizer": str(normalizer),
            "normalizer_sha256": _sha(normalizer),
            "evaluation_manifest": None,
            "action_dim": 20,
            "state_dim": 20,
            "camera_keys": ["global", "left_wrist", "right_wrist"],
        })
    payload = {
        "schema_version": 1,
        "release_id": f"shared-{count}-{condition}",
        "condition": condition,
        "tasks": tasks,
        "mixture_seed": 20260920,
        "model_seed": 20260824,
        "training_seed": 20260921,
        "optimizer": {
            "learning_rate": 5e-5,
            "final_learning_rate": 5e-6,
            "warmup_steps": 1,
            "batch_size": 1,
            "device_count": 1,
            "gradient_accumulation": 1,
        },
        "optimizer_steps": 12,
        "checkpoint_steps": [6, 12],
        "formal_training_authorized": False,
        "writable_paths": {"output": f"outputs/{condition}", "logs": f"logs/{condition}"},
    }
    path = tmp_path / f"mixture-{condition}.json"
    path.write_text(json.dumps(payload))
    return path


def _trainer_argv(tmp_path, mixture, stage, *, dataset=None):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    argv = [
        str(TRAINER),
        "--stage", stage.value,
        "--mixture-manifest", str(mixture),
        "--base-checkpoint", str(base),
        "--output", str(tmp_path / f"out-{stage.value}"),
        "--steps", "12",
        "--batch-size", "1",
        "--device-count", "1",
        "--learning-rate", "5e-5",
        "--final-learning-rate", "5e-6",
        "--warmup-steps", "1",
        "--model-seed", "20260824",
        "--training-seed", "20260921",
    ]
    if dataset is not None:
        argv.extend(["--dataset", str(dataset)])
    return argv


def _run_trainer_dry(trainer, monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", argv)
    trainer.main()
    output = capsys.readouterr().out
    plan, _ = json.JSONDecoder().raw_decode(output[output.index("{"):])
    assert "DRY_RUN_ONLY" in output
    return plan


@pytest.mark.parametrize("count", [3, 4])
def test_trainer_dry_run_records_shared_mixture_contract(
    trainer, tmp_path, monkeypatch, capsys, count
):
    mixture = _mixture(tmp_path, count=count, condition="full")
    plan = _run_trainer_dry(
        trainer,
        monkeypatch,
        capsys,
        _trainer_argv(tmp_path, mixture, StageName.PI_NATIVE_V2_SHARED_FULL),
    )
    assert len(plan["shared_mixture"]["contract"]["tasks"]) == count
    assert plan["shared_mixture"]["manifest_sha256"] == _sha(mixture)
    assert plan["shared_mixture"]["planned_task_counts"] == {
        task["name"]: 12 // count for task in plan["shared_mixture"]["contract"]["tasks"]
    }
    assert not (tmp_path / "out-pi_native_v2_shared_full").exists()


def test_trainer_rejects_dataset_with_mixture(trainer, tmp_path, monkeypatch):
    mixture = _mixture(tmp_path, count=3, condition="full")
    argv = _trainer_argv(
        tmp_path, mixture, StageName.PI_NATIVE_V2_SHARED_FULL, dataset=tmp_path
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer.main()


def test_trainer_rejects_non_shared_stage_with_mixture(trainer, tmp_path, monkeypatch):
    mixture = _mixture(tmp_path, count=3, condition="full")
    argv = _trainer_argv(
        tmp_path, mixture, StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="shared stage"):
        trainer.main()


def test_trainer_rejects_shared_stage_without_mixture(trainer, tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setattr(sys, "argv", [
        str(TRAINER), "--stage", StageName.PI_NATIVE_V2_SHARED_FULL.value,
        "--dataset", str(tmp_path), "--base-checkpoint", str(base),
        "--output", str(tmp_path / "out"), "--steps", "12", "--batch-size", "1",
        "--device-count", "1", "--warmup-steps", "1", "--model-seed", "20260824",
        "--training-seed", "20260921",
    ])
    with pytest.raises(ValueError, match="mixture manifest"):
        trainer.main()


def test_trainer_full_and_independent_dry_runs_are_matched(
    trainer, tmp_path, monkeypatch, capsys
):
    full_path = _mixture(tmp_path / "full", count=4, condition="full")
    independent_payload = json.loads(full_path.read_text())
    independent_payload["condition"] = "independent"
    independent_payload["release_id"] = "shared-4-independent"
    independent_payload["writable_paths"] = {
        "output": "outputs/independent", "logs": "logs/independent"
    }
    independent_path = tmp_path / "full" / "mixture-independent.json"
    independent_path.write_text(json.dumps(independent_payload))
    full_plan = _run_trainer_dry(
        trainer, monkeypatch, capsys,
        _trainer_argv(tmp_path / "full", full_path, StageName.PI_NATIVE_V2_SHARED_FULL),
    )
    independent_plan = _run_trainer_dry(
        trainer, monkeypatch, capsys,
        _trainer_argv(
            tmp_path / "full", independent_path, StageName.PI_NATIVE_V2_SHARED_INDEPENDENT
        ),
    )
    full = SharedMixtureSpec.from_json(full_path)
    independent = SharedMixtureSpec.from_json(independent_path)
    compare_matched_specs(full, independent)
    assert full_plan["shared_mixture"]["planned_task_counts"] == independent_plan[
        "shared_mixture"
    ]["planned_task_counts"]


@pytest.mark.parametrize("change", [None, "stage", "model_seed", "training_seed", "hash", "counts"])
def test_trainer_shared_resume_manifest_contract(trainer, tmp_path, change):
    mixture = _mixture(tmp_path, count=4, condition="full")
    spec = SharedMixtureSpec.from_json(mixture)
    mixture_sha = _sha(mixture)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    shared = trainer._shared_protocol(spec, manifest_sha256=mixture_sha, completed_steps=7)
    payload = {
        "stage": StageName.PI_NATIVE_V2_SHARED_FULL.value,
        "step": 7,
        "model_seed": 20260824,
        "training_seed": 20260921,
        "protocol": {"shared_mixture": shared},
    }
    if change == "stage":
        payload["stage"] = StageName.PI_NATIVE_V2_SHARED_INDEPENDENT.value
    elif change == "model_seed":
        payload["model_seed"] += 1
    elif change == "training_seed":
        payload["training_seed"] += 1
    elif change == "hash":
        payload["protocol"]["shared_mixture"]["manifest_sha256"] = "f" * 64
    elif change == "counts":
        payload["protocol"]["shared_mixture"]["task_counts_completed"][
            "aloha_handover_box"
        ] += 1
    (checkpoint / "manifest.json").write_text(json.dumps(payload))
    call = lambda: trainer._validate_shared_resume_manifest(
        checkpoint,
        stage=StageName.PI_NATIVE_V2_SHARED_FULL,
        spec=spec,
        manifest_sha256=mixture_sha,
        model_seed=20260824,
        training_seed=20260921,
    )
    if change is None:
        call()
    else:
        with pytest.raises(ValueError, match="shared resume"):
            call()


def test_trainer_checkpoint_protocol_records_counts_at_checkpoint_step(trainer, tmp_path):
    mixture = _mixture(tmp_path, count=3, condition="full")
    spec = SharedMixtureSpec.from_json(mixture)
    protocol = {"action_horizon": 50}
    saved = trainer._checkpoint_protocol(protocol, spec, _sha(mixture), step=8)
    counts = saved["shared_mixture"]["task_counts_completed"]
    assert set(counts) == set(MANDATORY)
    assert sum(counts.values()) == 8
    assert sorted(counts.values()) == [2, 3, 3]
    assert protocol == {"action_horizon": 50}


def test_trainer_stop_after_step_preserves_formal_optimizer_budget(
    trainer, tmp_path, monkeypatch, capsys
):
    mixture = _mixture(tmp_path, count=3, condition="full")
    argv = _trainer_argv(
        tmp_path, mixture, StageName.PI_NATIVE_V2_SHARED_FULL
    ) + ["--stop-after-step", "7"]
    plan = _run_trainer_dry(trainer, monkeypatch, capsys, argv)
    assert plan["optimizer_steps"] == 12
    assert plan["stop_after_step"] == 7


@pytest.mark.parametrize("stop", [0, 13])
def test_trainer_rejects_stop_outside_optimizer_budget(
    trainer, tmp_path, monkeypatch, stop
):
    mixture = _mixture(tmp_path, count=3, condition="full")
    argv = _trainer_argv(
        tmp_path, mixture, StageName.PI_NATIVE_V2_SHARED_FULL
    ) + ["--stop-after-step", str(stop)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="stop-after"):
        trainer.main()


def test_distributed_launcher_targets_canonical_pi05_layout():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'project_root / "scripts" / "train_dual_pi05.py"' in source
    assert 'project_root / "src"' in source
    assert "PI05_OPENPI_SRC" in source
    assert '"--distributed-process-count"' in source
    assert '"--distributed-process-id"' in source
