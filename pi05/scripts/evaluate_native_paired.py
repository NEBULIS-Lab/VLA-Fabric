#!/usr/bin/env python3
"""Paired Validation-200 rollouts for native pi0.5 Handover checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import signal
import time

from dm_env import StepType
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils

from pi05_fabric.agents.load_pi05 import load_dual_pi05
from pi05_fabric.agents.pi05_strong import native_trainable_filter
from pi05_fabric.data.aloha_dual_agent import merge_action
from pi05_fabric.data.aloha_dual_agent import split_observation
from pi05_fabric.data.converted_dataset import PairedTrainingSample
from pi05_fabric.data.pi05_batch import LocalNormalization
from pi05_fabric.data.pi05_batch import LocalQuantileNormalization
from pi05_fabric.data.pi05_batch import build_training_pair
from pi05_fabric.evaluation.gpu_claim import claim_jax_device
from pi05_fabric.evaluation.handover_validation import requested_condition_ids
from pi05_fabric.evaluation.handover_validation import summarize_candidate_rows
from pi05_fabric.evaluation.native_handover import diagnostic_validation_ids
from pi05_fabric.evaluation.native_handover import native_evaluation_spec
from pi05_fabric.evaluation.native_handover import native_inference_mode
from pi05_fabric.evaluation.native_handover import next_action_index
from pi05_fabric.evaluation.native_handover import shard_validation_ids
from pi05_fabric.evaluation.protocol import audit_rollout_rows
from pi05_fabric.evaluation.protocol import checkpoint_sha256
from pi05_fabric.training.audit import audit_training_checkpoint
from pi05_fabric.training.checkpoint import restore_training_state
from pi05_fabric.training.stages import StageName


class RolloutTimeout(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-stage",
        choices=(
            StageName.PI_NATIVE_RAW_COMMON_STAGE1.value,
            StageName.PI_NATIVE_THREE_PATH_STAGE2.value,
            StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1.value,
            StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2.value,
        ),
        required=True,
    )
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("aloha_handover_box", "aloha_shoes_table"),
        default="aloha_handover_box",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", default="validation")
    parser.add_argument("--validation-shard-index", type=int, default=0)
    parser.add_argument("--validation-shard-count", type=int, default=1)
    parser.add_argument("--allow-nonpaper-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--inference-mode", choices=("full", "core", "common_only"))
    parser.add_argument("--execute-horizon", type=int, choices=(5, 10, 20, 25))
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-parity", choices=("even", "odd"), default="even")
    parser.add_argument("--early-gpu-claim", action="store_true")
    parser.add_argument("--max-rollout-seconds", type=float, default=900.0)
    return parser.parse_args()


def _load_model(args, inference_mode):
    stage = StageName(args.expected_stage)
    spec = native_evaluation_spec(stage)
    manifest = audit_training_checkpoint(
        args.checkpoint,
        expected_stage=stage.value,
        expected_step=args.expected_step,
        verify_model=True,
    )
    snapshot = restore_training_state(args.checkpoint)
    protocol = manifest.get("protocol") or {}
    if protocol and int(protocol.get("action_horizon", spec.action_horizon)) != spec.action_horizon:
        raise ValueError("checkpoint action horizon does not match evaluation stage")
    if manifest.get("model_seed") != args.seed:
        raise ValueError(
            f"model seed mismatch: checkpoint={manifest.get('model_seed')} evaluation={args.seed}"
        )
    _, model = load_dual_pi05(
        args.base_checkpoint,
        seed=args.seed,
        mode=inference_mode,
        train_action_ffw=spec.train_action_ffw,
        train_action_attention=spec.train_action_attention,
        train_paligemma_kv=spec.train_paligemma_kv,
        action_horizon=spec.action_horizon,
    )
    selected = nnx.state(model).filter(
        native_trainable_filter(
            train_action_ffw=spec.train_action_ffw,
            train_action_attention=spec.train_action_attention,
            train_paligemma_kv=spec.train_paligemma_kv,
        )
    )
    selected.replace_by_pure_dict(snapshot.params)
    nnx.update(model, selected)
    del selected, snapshot
    gc.collect()
    return model, manifest, spec


def _pair_from_observation(observation, normalization, tokenizer, *, action_horizon):
    sample = {
        "images": {
            "global": np.asarray(observation["images"]["back"]),
            "left_wrist": np.asarray(observation["images"]["wrist_left"]),
            "right_wrist": np.asarray(observation["images"]["wrist_right"]),
        },
        "proprioception": np.asarray(observation["ee_6d_pos"], dtype=np.float32),
        "instruction": str(observation["language_instruction"]),
    }
    left, right = split_observation(sample)
    zeros = np.zeros((action_horizon, 10), dtype=np.float32)
    return build_training_pair(
        PairedTrainingSample(0, 0, left, right, zeros, zeros),
        normalization=normalization,
        tokenizer=tokenizer,
    )


def _run_rollout(
    env,
    condition_id,
    *,
    sampler,
    mode,
    normalization,
    tokenizer,
    seed,
    num_steps,
    execute_horizon,
    prediction_horizon,
    max_seconds,
):
    np.random.seed(seed + condition_id)
    timestep = env.reset()
    timestep = env.task.benchmark_init(env.physics, condition_id)
    started = time.monotonic()
    rewards = []
    plan_times = []
    plan_calls = 0
    actions = None
    action_index = 0
    ownership = None
    while True:
        if time.monotonic() - started > max_seconds:
            raise RolloutTimeout(f"rollout {condition_id} exceeded {max_seconds:.1f}s")
        if action_index == 0:
            pair = _pair_from_observation(
                timestep.observation, normalization, tokenizer, action_horizon=prediction_horizon
            )
            current_ownership = tuple(int(value) for value in pair.ownership[0])
            if ownership is None:
                ownership = current_ownership
            elif current_ownership != ownership:
                raise RuntimeError("ownership layout changed within one rollout")
            left_observation = jax.tree.map(jnp.asarray, pair.left_observation)
            right_observation = jax.tree.map(jnp.asarray, pair.right_observation)
            call_started = time.monotonic()
            left, right = sampler(
                jax.random.key(seed + condition_id * 1000 + plan_calls),
                left_observation,
                right_observation,
                ownership=ownership,
                mode=mode,
                num_steps=num_steps,
                preprocessed=False,
            )
            left, right = jax.device_get((left, right))
            plan_times.append(time.monotonic() - call_started)
            if not bool(np.isfinite(left).all() and np.isfinite(right).all()):
                raise FloatingPointError("non-finite action chunk")
            left, right = normalization.unnormalize_actions(left, right)
            actions = merge_action(left[0], right[0])
            plan_calls += 1
        timestep = env.step(actions[action_index])
        rewards.append(timestep.reward)
        action_index = next_action_index(
            action_index,
            execute_horizon=execute_horizon,
            prediction_horizon=prediction_horizon,
        )
        numeric = [value for value in rewards if value is not None]
        highest = float(max(numeric)) if numeric else 0.0
        if highest == env.task.max_reward or timestep.step_type == StepType.LAST:
            break
    return {
        "condition_id": condition_id,
        "success": int(highest == env.task.max_reward),
        "highest_reward": highest,
        "steps": len(rewards),
        "plan_calls": plan_calls,
        "mean_plan_seconds": float(np.mean(plan_times)) if plan_times else 0.0,
        "p95_plan_seconds": float(np.percentile(plan_times, 95)) if plan_times else 0.0,
        "episode_seconds": time.monotonic() - started,
        "finite_all": 1,
        "timeout": 0,
        "error": "",
    }


def _failure_row(condition_id: int, exc: Exception) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "success": 0,
        "highest_reward": 0.0,
        "steps": 0,
        "plan_calls": 0,
        "mean_plan_seconds": 0.0,
        "p95_plan_seconds": 0.0,
        "episode_seconds": 0.0,
        "finite_all": int(not isinstance(exc, FloatingPointError)),
        "timeout": int(isinstance(exc, RolloutTimeout)),
        "error": str(exc),
    }


def main() -> None:
    args = _parse_args()
    stage = StageName(args.expected_stage)
    spec = native_evaluation_spec(stage)
    inference_mode = (
        native_inference_mode(args.inference_mode, v2=stage in (
            StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1,
            StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        ))
        if args.inference_mode is not None
        else spec.mode
    )
    if args.diagnostic:
        if args.allow_nonpaper_smoke:
            raise ValueError("diagnostic and non-paper smoke modes are mutually exclusive")
        if stage not in (
            StageName.PI_NATIVE_THREE_PATH_STAGE2,
            StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        ):
            raise ValueError("no-retraining diagnostics require a native Stage2 checkpoint")
        if args.conditions != "validation":
            raise ValueError("diagnostic mode requires the validation condition set")
        diagnostic_ids = diagnostic_validation_ids(args.diagnostic_parity)
        ids = shard_validation_ids(
            diagnostic_ids,
            index=args.validation_shard_index,
            count=args.validation_shard_count,
        )
    else:
        ids = requested_condition_ids(
            args.conditions, allow_nonpaper_smoke=args.allow_nonpaper_smoke
        )
    if args.allow_nonpaper_smoke and args.validation_shard_count != 1:
        raise ValueError("non-paper smoke cannot be sharded")
    if not args.allow_nonpaper_smoke and not args.diagnostic:
        ids = shard_validation_ids(
            ids, index=args.validation_shard_index, count=args.validation_shard_count
        )
    claim_token = None
    if args.early_gpu_claim:
        claim_started = time.monotonic()
        claim_token = claim_jax_device()
        print(
            f"EARLY_GPU_CLAIM seconds={time.monotonic() - claim_started:.3f}",
            flush=True,
        )
    prediction_horizon = spec.action_horizon
    execute_horizon = spec.execution_horizon if args.execute_horizon is None else args.execute_horizon
    if execute_horizon > prediction_horizon:
        raise ValueError("execute horizon cannot exceed prediction horizon")
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    checkpoint_manifest = audit_training_checkpoint(
        args.checkpoint,
        expected_stage=stage.value,
        expected_step=args.expected_step,
        verify_model=True,
    )
    evaluation_manifest = {
        "schema_version": 1,
        "purpose": (
            "nonpaper_smoke"
            if args.allow_nonpaper_smoke
            else "fresh_test" if args.conditions == "fresh"
            else "no_retraining_diagnostic" if args.diagnostic else "development_validation"
        ),
        "formal_deployment_result": False,
        "conditions": list(ids),
        "validation_shard_index": args.validation_shard_index,
        "validation_shard_count": args.validation_shard_count,
        "diagnostic_parity": args.diagnostic_parity if args.diagnostic else None,
        "task": args.task,
        "stage": stage.value,
        "mode": inference_mode.value,
        "inference_mode": inference_mode.value,
        "checkpoint_training_mode": spec.mode.value,
        "channels": {
            **inference_mode.interaction_spec._asdict(),
            "remote_action_residual": inference_mode.remote_action_residual,
        },
        "prediction_horizon": prediction_horizon,
        "execute_horizon": execute_horizon,
        "num_denoising_steps": args.num_denoising_steps,
        "early_gpu_claim": args.early_gpu_claim,
        "seed": args.seed,
        "pair_environment_seed": True,
        "pair_initial_state": True,
        "pair_flow_noise": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_tree_sha256": checkpoint_sha256(args.checkpoint),
        "checkpoint_manifest": checkpoint_manifest,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n"
    )

    load_started = time.monotonic()
    model, checkpoint_manifest, spec = _load_model(args, inference_mode)
    print(
        f"MODEL_LOADED step={args.expected_step} seconds={time.monotonic() - load_started:.3f}",
        flush=True,
    )
    sampler = nnx_utils.module_jit(
        model.sample_actions,
        static_argnames=("ownership", "mode", "num_steps", "preprocessed"),
    )
    normalization = (
        LocalQuantileNormalization.from_json(args.dataset / "normalization_q01_q99.json")
        if stage in (
            StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1,
            StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        )
        else LocalNormalization.from_json(args.dataset / "normalization.json")
    )
    tokenizer = PaligemmaTokenizer(200)

    import tabletop

    env = tabletop.env(args.task, "ee_6d_pos")
    rows = []
    partial_path = args.output_dir / "rollouts.partial.csv"
    fields = [
        "condition_id",
        "success",
        "highest_reward",
        "steps",
        "plan_calls",
        "mean_plan_seconds",
        "p95_plan_seconds",
        "episode_seconds",
        "finite_all",
        "timeout",
        "error",
    ]
    with partial_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, condition_id in enumerate(ids, start=1):
            try:
                row = _run_rollout(
                    env,
                    condition_id,
                    sampler=sampler,
                    mode=inference_mode,
                    normalization=normalization,
                    tokenizer=tokenizer,
                    seed=args.seed,
                    num_steps=args.num_denoising_steps,
                    execute_horizon=execute_horizon,
                    prediction_horizon=prediction_horizon,
                    max_seconds=args.max_rollout_seconds,
                )
            except (RolloutTimeout, FloatingPointError, RuntimeError) as exc:
                row = _failure_row(condition_id, exc)
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            print(f"VALIDATION {index}/{len(ids)} {json.dumps(row, sort_keys=True)}", flush=True)

    if args.allow_nonpaper_smoke:
        successes = sum(int(row["success"]) for row in rows)
        result = {
            "status": "PASS" if all(int(row["finite_all"]) and not row["error"] for row in rows) else "FAIL",
            "purpose": "nonpaper_smoke",
            "successes": successes,
            "trials": len(rows),
        }
    elif args.diagnostic:
        audit_rollout_rows(rows, expected_ids=ids)
        successes = [row for row in rows if int(row["success"])]
        result = {
            "status": "COMPLETE",
            "purpose": "no_retraining_diagnostic",
            "stage": stage.value,
            "mode": inference_mode.value,
            "step": args.expected_step,
            "successes": len(successes),
            "trials": len(rows),
            "timeouts": sum(int(row["timeout"]) for row in rows),
            "errors": sum(bool(row["error"]) for row in rows),
            "mean_success_steps": (
                sum(float(row["steps"]) for row in successes) / len(successes)
                if successes
                else None
            ),
            "mean_plan_calls": sum(float(row["plan_calls"]) for row in rows) / len(rows),
        }
    elif args.validation_shard_count == 1:
        candidate = summarize_candidate_rows(
            step=args.expected_step, rows=rows, expected_ids=ids
        )
        result = {
            "status": "COMPLETE",
            "purpose": "fresh_test" if args.conditions == "fresh" else "development_validation",
            "stage": stage.value,
            "mode": inference_mode.value,
            **candidate.__dict__,
        }
    else:
        successes = [row for row in rows if int(row["success"])]
        result = {
            "status": "COMPLETE",
            "purpose": "fresh_test_shard" if args.conditions == "fresh" else "development_validation_shard",
            "stage": stage.value,
            "mode": inference_mode.value,
            "step": args.expected_step,
            "shard_index": args.validation_shard_index,
            "shard_count": args.validation_shard_count,
            "successes": len(successes),
            "trials": len(rows),
            "timeouts": sum(int(row["timeout"]) for row in rows),
            "errors": sum(bool(row["error"]) for row in rows),
            "mean_success_steps": (
                sum(float(row["steps"]) for row in successes) / len(successes)
                if successes
                else None
            ),
        }
    result["checkpoint_manifest"] = checkpoint_manifest
    result["rows"] = rows
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    partial_path.rename(args.output_dir / "rollouts.csv")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
