#!/usr/bin/env python3
"""Train one frozen experiment2 stage; dry-run unless --execute is supplied."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer

from pi05_fabric.agents.load_pi05 import load_dual_pi05
from pi05_fabric.data.batch_stack import stack_training_pairs
from pi05_fabric.data.converted_dataset import ConvertedAlohaDataset
from pi05_fabric.data.shared_mixture import SharedMixtureSpec
from pi05_fabric.data.shared_mixture import task_counts_through_step
from pi05_fabric.data.shared_mixture import task_index_for_step
from pi05_fabric.data.pi05_batch import LocalNormalization
from pi05_fabric.data.pi05_batch import LocalQuantileNormalization
from pi05_fabric.data.pi05_batch import Pi05TrainingPair
from pi05_fabric.data.pi05_batch import build_training_pair
from pi05_fabric.training.checkpoint import restore_training_state
from pi05_fabric.training.distributed import process_sample_indices
from pi05_fabric.training.distributed import validate_replica_topology
from pi05_fabric.training.engine import create_engine
from pi05_fabric.training.engine import create_native_engine
from pi05_fabric.training.engine_checkpoint import apply_same_stage_snapshot
from pi05_fabric.training.engine_checkpoint import fork_from_stage1
from pi05_fabric.training.engine_checkpoint import write_engine_checkpoint
from pi05_fabric.training.launch_config import LaunchConfig
from pi05_fabric.training.optimizer import OptimizerSettings
from pi05_fabric.training.optimizer import GroupedOptimizerSettings
from pi05_fabric.training.optimizer import create_optimizer
from pi05_fabric.training.protocol import data_rng_for_sample
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import parameter_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=StageName, required=True, choices=list(StageName))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--mixture-manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("external/pi05_base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True, help="Per-device batch size")
    parser.add_argument("--device-count", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--final-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--parent-training-seed", type=int)
    parser.add_argument("--freeze-action-expert-ffw", action="store_true")
    parser.add_argument("--freeze-action-expert-attention", action="store_true")
    parser.add_argument("--freeze-paligemma-base-kv", action="store_true")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--distributed-process-count", type=int, default=1)
    parser.add_argument("--distributed-process-id", type=int, default=0)
    parser.add_argument("--distributed-coordinator-address")
    parser.add_argument("--preload-dataset", action="store_true")
    parser.add_argument("--preload-workers", type=int, default=1)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Gate-only stop that preserves the configured optimizer schedule",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


_SHARED_STAGES = {
    StageName.PI_NATIVE_V2_SHARED_FULL: "full",
    StageName.PI_NATIVE_V2_SHARED_INDEPENDENT: "independent",
}


def _prepare_data_contract(args: argparse.Namespace) -> tuple[SharedMixtureSpec | None, str | None]:
    is_shared = args.stage in _SHARED_STAGES
    if args.dataset is not None and args.mixture_manifest is not None:
        raise ValueError("dataset and mixture manifest are mutually exclusive")
    if is_shared and args.mixture_manifest is None:
        raise ValueError("a shared stage requires a mixture manifest")
    if not is_shared and args.mixture_manifest is not None:
        raise ValueError("a mixture manifest requires a shared stage")
    if not is_shared:
        if args.dataset is None:
            args.dataset = Path("data/converted/aloha_handover_box")
        return None, None

    spec = SharedMixtureSpec.from_json(args.mixture_manifest)
    spec.validate_files()
    manifest_options = json.loads(args.mixture_manifest.read_text(encoding="utf-8"))
    if bool(manifest_options.get("freeze_action_expert_ffw", False)) != args.freeze_action_expert_ffw:
        raise ValueError("Shared action-expert FFW freeze flag differs from frozen manifest")
    expected_condition = _SHARED_STAGES[args.stage]
    if spec.condition != expected_condition:
        raise ValueError(
            f"shared stage requires condition {expected_condition}, found {spec.condition}"
        )
    args.dataset = spec.tasks[0].dataset
    return spec, hashlib.sha256(args.mixture_manifest.read_bytes()).hexdigest()


def _validate_shared_launch(
    spec: SharedMixtureSpec, args: argparse.Namespace, config: LaunchConfig
) -> None:
    if spec.model_seed != config.resolved_model_seed:
        raise ValueError("mixture model_seed differs from launch")
    if spec.training_seed != config.resolved_training_seed:
        raise ValueError("mixture training_seed differs from launch")
    if spec.optimizer_steps is not None and spec.optimizer_steps != config.steps:
        raise ValueError("mixture optimizer_steps differs from launch")
    expected = {
        "learning_rate": config.learning_rate,
        "final_learning_rate": (
            config.learning_rate
            if config.final_learning_rate is None
            else config.final_learning_rate
        ),
        "warmup_steps": config.warmup_steps,
        "batch_size": config.batch_size,
        "device_count": config.device_count,
        "gradient_accumulation": config.gradient_accumulation,
    }
    for key, value in expected.items():
        if spec.optimizer.get(key) != value:
            raise ValueError(f"mixture optimizer {key} differs from launch")


def _shared_protocol(
    spec: SharedMixtureSpec, *, manifest_sha256: str, completed_steps: int
) -> dict[str, object]:
    names = tuple(task.name for task in spec.tasks)
    return {
        "manifest_sha256": manifest_sha256,
        "contract": spec.contract_dict(),
        "task_counts_completed": task_counts_through_step(
            names, seed=spec.mixture_seed, completed_steps=completed_steps
        ),
    }


def _checkpoint_protocol(
    protocol: dict, spec: SharedMixtureSpec | None, manifest_sha256: str | None, step: int
) -> dict:
    result = copy.deepcopy(protocol)
    if spec is not None:
        result["shared_mixture"] = _shared_protocol(
            spec, manifest_sha256=manifest_sha256, completed_steps=step
        )
    return result


def _validate_shared_resume_manifest(
    checkpoint: Path,
    *,
    stage: StageName,
    spec: SharedMixtureSpec,
    manifest_sha256: str,
    model_seed: int,
    training_seed: int,
) -> None:
    saved = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if saved.get("stage") != stage.value:
        raise ValueError("shared resume stage differs from checkpoint")
    if saved.get("model_seed") != model_seed or saved.get("training_seed") != training_seed:
        raise ValueError("shared resume seeds differ from checkpoint")
    step = int(saved.get("step", -1))
    expected = _shared_protocol(spec, manifest_sha256=manifest_sha256, completed_steps=step)
    observed = (saved.get("protocol") or {}).get("shared_mixture")
    if observed != expected:
        raise ValueError("shared resume mixture contract or task counters differ")


def _initialize_distributed(args: argparse.Namespace) -> bool:
    if args.distributed_process_count == 1:
        if args.distributed_process_id != 0 or args.distributed_coordinator_address is not None:
            raise ValueError("single-process mode cannot set a distributed rank or coordinator")
        return False
    if args.distributed_coordinator_address is None:
        raise ValueError("distributed_coordinator_address is required for multi-process training")
    if not 0 <= args.distributed_process_id < args.distributed_process_count:
        raise ValueError("distributed_process_id is outside the distributed world")
    jax.distributed.initialize(
        coordinator_address=args.distributed_coordinator_address,
        num_processes=args.distributed_process_count,
        process_id=args.distributed_process_id,
        local_device_ids=0,
    )
    return True


def _launch_config(args: argparse.Namespace) -> LaunchConfig:
    return LaunchConfig(
        stage=args.stage,
        dataset=args.dataset,
        base_checkpoint=args.base_checkpoint,
        output=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        warmup_steps=args.warmup_steps,
        device_count=args.device_count,
        gradient_accumulation=args.gradient_accumulation,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        model_seed=args.model_seed,
        training_seed=args.training_seed,
        parent_training_seed=args.parent_training_seed,
        stage1_checkpoint=args.stage1_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
    )


def _device_pair(pair):
    return Pi05TrainingPair(
        jax.tree.map(jnp.asarray, pair.left_observation),
        jax.tree.map(jnp.asarray, pair.right_observation),
        jnp.asarray(pair.left_actions),
        jnp.asarray(pair.right_actions),
        jnp.asarray(pair.ownership),
    )


def _host_scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def _sync(name: str, *, distributed: bool) -> None:
    if distributed:
        multihost_utils.sync_global_devices(name)


def main() -> None:
    args = _parse_args()
    shared_spec, mixture_manifest_sha256 = _prepare_data_contract(args)
    distributed = _initialize_distributed(args) if args.execute else False
    process_index = jax.process_index()
    primary = process_index == 0
    print(
        f"rank={process_index} process_pid={os.getpid()} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )
    config = _launch_config(args)
    config.validate_paths()
    if args.stop_after_step is not None and not 0 < args.stop_after_step <= config.steps:
        raise ValueError("stop-after-step must be within the optimizer budget")
    if shared_spec is not None:
        _validate_shared_launch(shared_spec, args, config)
    selected_devices = tuple(jax.local_devices())
    action_horizon = 50 if config.mode.is_v2 else 20
    execution_horizon = 25 if config.mode.is_v2 else 10
    normalization_path = None
    normalization_payload = None
    if shared_spec is None:
        normalization_path = config.dataset / (
            "normalization_q01_q99.json" if config.mode.is_v2 else "normalization.json"
        )
        normalization_payload = json.loads(normalization_path.read_text())
        if config.mode.is_v2:
            manifest_bytes = (config.dataset / "manifest.json").read_bytes()
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            if normalization_payload.get("dataset_manifest_sha256") != manifest_sha:
                raise ValueError("V2 quantile statistics do not match the dataset manifest")
            if normalization_payload.get("schema_version") != 2:
                raise ValueError("V2 requires q01/q99 normalization schema 2")
    protocol_metadata = {
        "action_horizon": action_horizon,
        "execution_horizon": execution_horizon,
        "flow_steps": 10,
        "normalization_file": None if normalization_path is None else normalization_path.name,
        "normalization": normalization_payload,
    }
    if shared_spec is not None:
        protocol_metadata["shared_mixture"] = _shared_protocol(
            shared_spec,
            manifest_sha256=mixture_manifest_sha256,
            completed_steps=0,
        )
    final_lr = config.learning_rate if config.final_learning_rate is None else config.final_learning_rate
    optimizer_settings = OptimizerSettings(
        peak_learning_rate=config.learning_rate,
        final_learning_rate=final_lr,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
    )
    checkpoint_steps = (
        shared_spec.checkpoint_steps
        if shared_spec is not None and shared_spec.checkpoint_steps
        else config.checkpoint_steps
    )
    plan = {
        "stage": config.stage.value,
        "mode": config.mode.value,
        "dataset": str(config.dataset.resolve()),
        "base_checkpoint": str(config.base_checkpoint.resolve()),
        "output": str(config.output.resolve()),
        "optimizer_steps": config.steps,
        "stop_after_step": args.stop_after_step,
        "per_device_batch_size": config.batch_size,
        "global_batch_size": config.global_batch_size,
        "device_count": config.device_count,
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "gradient_accumulation": config.gradient_accumulation,
        "optimizer": {
            "name": "adamw",
            "peak_learning_rate": config.learning_rate,
            "final_learning_rate": final_lr,
            "warmup_steps": config.warmup_steps,
            "beta1": optimizer_settings.beta1,
            "beta2": optimizer_settings.beta2,
            "epsilon": optimizer_settings.epsilon,
            "weight_decay": optimizer_settings.weight_decay,
            "clip_gradient_norm": optimizer_settings.clip_gradient_norm,
            "scheduler": "warmup_cosine",
        },
        "checkpoint_every": config.checkpoint_every,
        "checkpoint_steps": list(checkpoint_steps),
        "model_seed": config.resolved_model_seed,
        "training_seed": config.resolved_training_seed,
        "parent_training_seed": config.parent_training_seed,
        "train_action_expert_ffw": not args.freeze_action_expert_ffw,
        "train_action_expert_attention": not args.freeze_action_expert_attention,
        "train_paligemma_base_kv": not args.freeze_paligemma_base_kv,
        "stage1_checkpoint": None if config.stage1_checkpoint is None else str(config.stage1_checkpoint.resolve()),
        "resume_checkpoint": None if config.resume_checkpoint is None else str(config.resume_checkpoint.resolve()),
        "topology": "one_process_per_gpu_replicated_dual_agent_graph",
        "preload_dataset": args.preload_dataset,
        "preload_workers": args.preload_workers,
        "protocol": protocol_metadata,
    }
    if shared_spec is not None:
        plan["mixture_manifest"] = str(args.mixture_manifest.resolve())
        plan["shared_mixture"] = {
            "manifest_sha256": mixture_manifest_sha256,
            "contract": shared_spec.contract_dict(),
            "planned_task_counts": task_counts_through_step(
                tuple(task.name for task in shared_spec.tasks),
                seed=shared_spec.mixture_seed,
                completed_steps=config.steps,
            ),
        }
    if config.mode.is_native:
        plan["optimizer"] = {
            "name": "grouped_adamw",
            "scheduler": "warmup_cosine",
            "warmup_steps": config.warmup_steps,
            "beta1": optimizer_settings.beta1,
            "beta2": optimizer_settings.beta2,
            "epsilon": optimizer_settings.epsilon,
            "weight_decay": optimizer_settings.weight_decay,
            "global_clip_gradient_norm": optimizer_settings.clip_gradient_norm,
            "groups": {
                "action_expert_base": {
                    "peak_learning_rate": 5e-6,
                    "final_learning_rate": 5e-7,
                },
                "paligemma_kv_norm": {
                    "peak_learning_rate": 3e-6,
                    "final_learning_rate": 3e-7,
                },
                "lora_and_projections": {
                    "peak_learning_rate": config.learning_rate,
                    "final_learning_rate": final_lr,
                },
            },
        }
    if primary:
        print(json.dumps(plan, indent=2))
    if not args.execute:
        if primary:
            print("DRY_RUN_ONLY: add --execute to start training")
        if distributed:
            jax.distributed.shutdown()
        return

    if shared_spec is not None and args.resume_checkpoint is not None:
        _validate_shared_resume_manifest(
            args.resume_checkpoint,
            stage=config.stage,
            spec=shared_spec,
            manifest_sha256=mixture_manifest_sha256,
            model_seed=config.resolved_model_seed,
            training_seed=config.resolved_training_seed,
        )

    validate_replica_topology(
        requested_device_count=config.device_count,
        process_count=jax.process_count(),
        local_device_count=jax.local_device_count(),
    )

    _sync("validated_launch", distributed=distributed)
    if primary:
        config.output.mkdir(parents=True, exist_ok=config.resume_checkpoint is not None)
        manifest_name = "resume_launch_manifest.json" if config.resume_checkpoint is not None else "launch_manifest.json"
        (config.output / manifest_name).write_text(json.dumps(plan, indent=2) + "\n")
    _sync("created_output", distributed=distributed)

    normalization = None
    normalizations = None
    if shared_spec is None:
        normalization = (
            LocalQuantileNormalization.from_json(normalization_path)
            if config.mode.is_v2
            else LocalNormalization.from_json(normalization_path)
        )
    else:
        normalizations = {
            task.name: LocalQuantileNormalization.from_json(task.normalizer)
            for task in shared_spec.tasks
        }
    tokenizer = PaligemmaTokenizer(200)

    load_started = time.monotonic()
    _, model = load_dual_pi05(
        config.base_checkpoint,
        seed=config.resolved_model_seed,
        mode=config.mode,
        train_action_ffw=not args.freeze_action_expert_ffw,
        train_action_attention=not args.freeze_action_expert_attention,
        train_paligemma_kv=not args.freeze_paligemma_base_kv,
        action_horizon=action_horizon,
    )
    load_seconds = time.monotonic() - load_started
    _sync("loaded_models", distributed=distributed)
    print(f"rank={process_index} dual_model_load_seconds={load_seconds:.3f}")
    if config.mode.is_native:
        grouped_settings = GroupedOptimizerSettings(
            total_steps=config.steps,
            warmup_steps=config.warmup_steps,
            action_expert_peak=5e-6,
            action_expert_final=5e-7,
            paligemma_peak=3e-6,
            paligemma_final=3e-7,
            adapter_peak=config.learning_rate,
            adapter_final=final_lr,
        )
        engine = create_native_engine(
            model,
            mode=config.mode,
            settings=grouped_settings,
            train_action_ffw=not args.freeze_action_expert_ffw,
            train_action_attention=not args.freeze_action_expert_attention,
            train_paligemma_kv=not args.freeze_paligemma_base_kv,
        )
    else:
        engine = create_engine(
            model,
            mode=config.mode,
            tx=create_optimizer(optimizer_settings),
        )
    del model
    gc.collect()
    rng = jax.random.key(config.resolved_training_seed)
    parent_hash = None
    if config.stage1_checkpoint is not None:
        stage1 = restore_training_state(config.stage1_checkpoint)
        if (
            config.mode.is_native
            and stage1.training_seed is not None
            and stage1.training_seed == config.resolved_training_seed
        ):
            raise ValueError("Stage 2 training seed must differ from Stage 1")
        forked = fork_from_stage1(
            engine,
            stage1,
            target=config.stage,
            rng=rng,
            model_seed=config.resolved_model_seed,
            training_seed=config.resolved_training_seed,
            protocol_metadata=protocol_metadata,
        )
        parent_hash = forked.parent_model_sha256
        del stage1
        del forked
        gc.collect()
    elif config.resume_checkpoint is not None:
        resumed = restore_training_state(config.resume_checkpoint)
        apply_same_stage_snapshot(engine, resumed, expected_stage=config.stage)
        rng = resumed.rng
        parent_hash = resumed.parent_model_sha256
        del resumed
        gc.collect()

    if config.device_count > 1:
        engine.enable_data_parallel(selected_devices)
    _sync("allocated_training_state", distributed=distributed)

    dataset_started = time.monotonic()
    datasets = None
    if shared_spec is None:
        dataset = ConvertedAlohaDataset(
            config.dataset,
            action_horizon=action_horizon,
            preload=args.preload_dataset,
            preload_workers=args.preload_workers,
        )
        preloaded_bytes = dataset.preloaded_bytes
    else:
        datasets = {
            task.name: ConvertedAlohaDataset(
                task.dataset,
                action_horizon=action_horizon,
                preload=args.preload_dataset,
                preload_workers=args.preload_workers,
            )
            for task in shared_spec.tasks
        }
        preloaded_bytes = sum(item.preloaded_bytes for item in datasets.values())
    dataset_seconds = time.monotonic() - dataset_started
    _sync("loaded_dataset", distributed=distributed)
    print(
        f"rank={process_index} dataset_load_seconds={dataset_seconds:.3f} "
        f"preloaded_gib={preloaded_bytes / 2**30:.3f}"
    )
    checkpoint_targets = set(checkpoint_steps)
    samples_per_process = (
        config.batch_size
        * jax.local_device_count()
        * config.gradient_accumulation
    )
    sample_indices = process_sample_indices(
        process_index=process_index,
        samples_per_process=samples_per_process,
    )
    run_until_step = config.steps if args.stop_after_step is None else args.stop_after_step
    for optimizer_step in range(engine.current_step(), run_until_step):
        step_started = time.monotonic()
        if shared_spec is not None:
            task_index = task_index_for_step(
                len(shared_spec.tasks), seed=shared_spec.mixture_seed, step=optimizer_step
            )
            task_name = shared_spec.tasks[task_index].name
            dataset = datasets[task_name]
            normalization = normalizations[task_name]
            if primary:
                print(f"shared_task step={optimizer_step + 1} task={task_name}")
        pairs = []
        for sample_index in sample_indices:
            data_rng = data_rng_for_sample(
                seed=config.resolved_training_seed,
                step=optimizer_step,
                sample_index=sample_index,
            )
            episode_id, episode_step = dataset.sample_index(data_rng)
            pairs.append(
                build_training_pair(
                    dataset.get(episode_id, episode_step),
                    normalization=normalization,
                    tokenizer=tokenizer,
                )
            )
        pair = _device_pair(stack_training_pairs(pairs))
        rng, step_rng = jax.random.split(rng)
        engine.state, info = engine.step(
            engine.state,
            step_rng,
            pair.left_observation,
            pair.right_observation,
            pair.left_actions,
            pair.right_actions,
            pair.ownership,
            gradient_accumulation=config.gradient_accumulation,
        )
        loss = _host_scalar(info.loss)
        left_loss = _host_scalar(info.left_loss)
        right_loss = _host_scalar(info.right_loss)
        grad_norm = _host_scalar(info.grad_norm)
        local_loss = _host_scalar(info.local_loss)
        local_left_loss = _host_scalar(info.local_left_loss)
        local_right_loss = _host_scalar(info.local_right_loss)
        local_grad_norm = _host_scalar(info.local_grad_norm)
        finite = bool(np.asarray(jax.device_get(info.finite)).all())
        current_step = engine.current_step()
        step_seconds = time.monotonic() - step_started
        print(
            f"rank_step={current_step} local_loss={local_loss:.6f} "
            f"local_left_loss={local_left_loss:.6f} "
            f"local_right_loss={local_right_loss:.6f} "
            f"local_grad_norm={local_grad_norm:.6f}"
        )
        if not finite:
            print(f"rank={process_index} non_finite_local_step={current_step}")
            raise FloatingPointError(f"non-finite step rejected at step {current_step}")
        if primary:
            print(
                f"step={current_step} loss={loss:.6f} "
                f"left_loss={left_loss:.6f} right_loss={right_loss:.6f} "
                f"grad_norm={grad_norm:.6f} step_seconds={step_seconds:.3f}"
            )

        if current_step in checkpoint_targets:
            if primary:
                checkpoint_started = time.monotonic()
                checkpoint = config.output / "checkpoints" / f"step_{current_step:08d}"
                write_engine_checkpoint(
                    checkpoint,
                    engine,
                    stage=config.stage,
                    rng=rng,
                    parent_model_sha256=parent_hash,
                    model_seed=config.resolved_model_seed,
                    training_seed=config.resolved_training_seed,
                    protocol_metadata=_checkpoint_protocol(
                        protocol_metadata,
                        shared_spec,
                        mixture_manifest_sha256,
                        current_step,
                    ),
                )
                print(
                    f"CHECKPOINT={checkpoint} "
                    f"save_seconds={time.monotonic() - checkpoint_started:.3f}"
                )
            _sync(f"checkpoint_{current_step}", distributed=distributed)

    final_step = engine.current_step()
    final_state = engine.host_state()
    final_digest = parameter_sha256(engine.selected_params(final_state).to_pure_dict())
    if distributed:
        multihost_utils.assert_equal(
            np.frombuffer(bytes.fromhex(final_digest), dtype=np.uint8),
            "final trainable parameter digests differ across ranks",
        )
    print(f"rank={process_index} final_step={final_step} final_model_sha256={final_digest}")
    memory_stats = selected_devices[0].memory_stats() or {}
    print(
        "JAX_MEMORY_STATS "
        + json.dumps(
            {
                key: int(value)
                for key, value in memory_stats.items()
                if isinstance(value, (int, np.integer))
            },
            sort_keys=True,
        )
    )
    final_checkpoint = config.output / "checkpoints" / f"step_{final_step:08d}"
    if primary and not final_checkpoint.exists():
        write_engine_checkpoint(
            final_checkpoint,
            engine,
            stage=config.stage,
            rng=rng,
            parent_model_sha256=parent_hash,
            model_seed=config.resolved_model_seed,
            training_seed=config.resolved_training_seed,
            protocol_metadata=_checkpoint_protocol(
                protocol_metadata,
                shared_spec,
                mixture_manifest_sha256,
                final_step,
            ),
        )
    _sync("final_checkpoint", distributed=distributed)
    if primary:
        print(f"FINAL_CHECKPOINT={final_checkpoint}")
    if distributed:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
