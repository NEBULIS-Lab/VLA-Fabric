#!/usr/bin/env python3
"""Train one audited three- or four-agent frame stage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
import time

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer

from pi05_fabric.agents.load_pi05 import load_local_role_pi05
from pi05_fabric.agents.multiagent_pi05 import MultiAgentPi05Mode
from pi05_fabric.data.multiarm_frame_tasks import MultiArmQuantileNormalization
from pi05_fabric.data.multiarm_frame_tasks import MultiArmTaskDataset
from pi05_fabric.data.four_arm_pi05 import build_role_training_sample
from pi05_fabric.data.four_arm_pi05 import stack_role_training_samples
from pi05_fabric.training.agent_parallel import AgentParallelTopology
from pi05_fabric.training.agent_parallel import batch_configuration
from pi05_fabric.training.agent_parallel import role_rng
from pi05_fabric.training.multiagent_checkpoint import finalize_team_checkpoint
from pi05_fabric.training.multiagent_checkpoint import fork_role_engine_from_stage1
from pi05_fabric.training.multiagent_checkpoint import restore_role_snapshot
from pi05_fabric.training.multiagent_checkpoint import resume_role_engine
from pi05_fabric.training.multiagent_checkpoint import save_role_checkpoint
from pi05_fabric.training.multiagent_checkpoint import snapshot_role_engine
from pi05_fabric.training.multiagent_engine import create_multiagent_engine
from pi05_fabric.training.optimizer import GroupedOptimizerSettings
from pi05_fabric.training.protocol import checkpoint_plan
from pi05_fabric.training.protocol import checkpoint_steps
from pi05_fabric.training.protocol import data_rng_for_sample
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import parameter_sha256


STAGES = (
    StageName.PI05_FRAME4_INDEPENDENT_DIRECT,
    StageName.PI05_FRAME3_COMMON_STAGE1,
    StageName.PI05_FRAME3_FULL_STAGE2,
    StageName.PI05_FRAME3_FULL_DIRECT,
    StageName.PI05_FRAME3_INDEPENDENT_DIRECT,
)

STAGE_AGENT_COUNT = {
    StageName.PI05_FRAME4_INDEPENDENT_DIRECT: 4,
    StageName.PI05_FRAME3_COMMON_STAGE1: 3,
    StageName.PI05_FRAME3_FULL_STAGE2: 3,
    StageName.PI05_FRAME3_FULL_DIRECT: 3,
    StageName.PI05_FRAME3_INDEPENDENT_DIRECT: 3,
}

STAGE_MODE = {
    StageName.PI05_FRAME4_INDEPENDENT_DIRECT: MultiAgentPi05Mode.INDEPENDENT,
    StageName.PI05_FRAME3_COMMON_STAGE1: MultiAgentPi05Mode.COMMON_ONLY,
    StageName.PI05_FRAME3_FULL_STAGE2: MultiAgentPi05Mode.FULL,
    StageName.PI05_FRAME3_FULL_DIRECT: MultiAgentPi05Mode.FULL,
    StageName.PI05_FRAME3_INDEPENDENT_DIRECT: MultiAgentPi05Mode.INDEPENDENT,
}

STAGE_STEPS = {
    StageName.PI05_FRAME4_INDEPENDENT_DIRECT: 30_000,
    StageName.PI05_FRAME3_COMMON_STAGE1: 10_000,
    StageName.PI05_FRAME3_FULL_STAGE2: 20_000,
    StageName.PI05_FRAME3_FULL_DIRECT: 30_000,
    StageName.PI05_FRAME3_INDEPENDENT_DIRECT: 30_000,
}

STAGE_PARENT = {
    StageName.PI05_FRAME3_FULL_STAGE2: StageName.PI05_FRAME3_COMMON_STAGE1,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=StageName, choices=STAGES, required=True)
    parser.add_argument(
        "--task",
        choices=("frame4_insertion", "frame3_triangle_insertion"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quantiles", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--normalization-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--team-microbatch", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument("--recovery-checkpoint-every", type=int, default=0)
    parser.add_argument("--action-expert-peak", type=float, default=5e-6)
    parser.add_argument("--action-expert-final", type=float, default=5e-7)
    parser.add_argument("--paligemma-peak", type=float, default=3e-6)
    parser.add_argument("--paligemma-final", type=float, default=3e-7)
    parser.add_argument("--adapter-peak", type=float, default=5e-5)
    parser.add_argument("--adapter-final", type=float, default=5e-6)
    parser.add_argument("--model-seed", type=int, default=731)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--distributed-process-count", type=int, required=True)
    parser.add_argument("--distributed-process-id", type=int, default=0)
    parser.add_argument("--distributed-coordinator-address")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--allow-nonformal-steps", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def initialize_distributed(args) -> bool:
    expected = STAGE_AGENT_COUNT[args.stage]
    if args.distributed_process_count != expected:
        raise ValueError(f"{args.stage.value} requires exactly {expected} processes")
    if args.distributed_coordinator_address is None:
        raise ValueError("distributed coordinator address is required")
    jax.distributed.initialize(
        coordinator_address=args.distributed_coordinator_address,
        num_processes=args.distributed_process_count,
        process_id=args.distributed_process_id,
        local_device_ids=0,
    )
    return True


def sync(name: str) -> None:
    multihost_utils.sync_global_devices(name)


def host_scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def device_sample(sample):
    return (
        jax.tree.map(jnp.asarray, sample.observation),
        jnp.asarray(sample.actions),
        jnp.asarray(sample.ownership),
    )


def protocol_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def recorded_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_sha256(path: Path, supplied: str | None) -> str:
    if supplied is None:
        return recorded_file_sha256(path)
    if re.fullmatch(r"[0-9a-f]{64}", supplied) is None:
        raise ValueError(f"invalid supplied SHA-256 for {path}")
    return supplied


def parse_permanent_checkpoint_steps(args) -> tuple[int, ...]:
    if not args.checkpoint_steps:
        return checkpoint_steps(total_steps=args.steps, every=args.checkpoint_every)
    try:
        return tuple(
            int(value.strip())
            for value in args.checkpoint_steps.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ValueError("checkpoint steps must be comma-separated integers") from error


def validate_args(args, topology):
    expected_agent_count = STAGE_AGENT_COUNT[args.stage]
    if topology.world_size != expected_agent_count:
        raise ValueError(
            f"the formal {expected_agent_count}-arm run requires exactly "
            f"{expected_agent_count} ranks"
        )
    if not args.dataset.is_file() or not args.quantiles.is_file():
        raise FileNotFoundError("audited multi-arm data or quantiles are missing")
    if not args.base_checkpoint.exists():
        raise FileNotFoundError("PI0.5 base checkpoint is missing")
    if args.steps <= 0 or args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("invalid step or warmup configuration")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        raise ValueError("stop_after must be within the schedule")
    if args.recovery_checkpoint_every < 0:
        raise ValueError("recovery checkpoint interval cannot be negative")
    learning_rate_pairs = (
        (args.action_expert_peak, args.action_expert_final),
        (args.paligemma_peak, args.paligemma_final),
        (args.adapter_peak, args.adapter_final),
    )
    if any(final <= 0 or peak <= 0 or final > peak for peak, final in learning_rate_pairs):
        raise ValueError("invalid grouped learning-rate boundaries")
    if not args.allow_nonformal_steps and args.steps != STAGE_STEPS[args.stage]:
        raise ValueError("formal stage length does not match the frozen protocol")
    if args.stage not in STAGE_PARENT:
        if args.stage1_checkpoint is not None:
            raise ValueError("base-initialized stages cannot load a Stage 1 parent")
    elif args.stage1_checkpoint is None and args.resume_checkpoint is None:
        raise ValueError("Stage 2 requires a Stage 1 team checkpoint")
    if args.stage1_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("stage fork and same-stage resume are mutually exclusive")
    batch = batch_configuration(
        team_microbatch=args.team_microbatch,
        accumulation=args.gradient_accumulation,
        data_parallel_teams=topology.team_count,
    )
    if not args.allow_nonformal_steps and batch.global_team_batch != 18:
        raise ValueError("formal global team batch must be 18")
    return batch


def main():
    args = parse_args()
    agent_count = STAGE_AGENT_COUNT[args.stage]
    topology = AgentParallelTopology.create(
        world_size=args.distributed_process_count,
        agent_count=agent_count,
    )
    batch = validate_args(args, topology)
    mode = STAGE_MODE[args.stage]
    permanent_checkpoint_steps = parse_permanent_checkpoint_steps(args)
    checkpoint_schedule = checkpoint_plan(
        total_steps=args.steps,
        permanent_steps=permanent_checkpoint_steps,
        recovery_every=args.recovery_checkpoint_every,
    )
    shared_protocol = {
        "schema_version": 1,
        "task": args.task,
        "instruction": args.instruction,
        "agent_count": agent_count,
        "mode": mode.value,
        "action_horizon": 50,
        "execution_horizon": 25,
        "flow_steps": 10,
        "flow_noise": "role_local_deterministic",
        "normalization": "q01_q99_per_role",
        "dataset_sha256": resolved_sha256(args.dataset, args.dataset_sha256),
        "normalization_sha256": resolved_sha256(
            args.quantiles, args.normalization_sha256
        ),
        "global_team_batch": batch.global_team_batch,
        "team_microbatch": batch.team_microbatch,
        "gradient_accumulation": batch.accumulation,
        "train_action_expert_ffw_base": False,
        "train_action_expert_attention": True,
        "train_paligemma_kv": True,
        "remote_action_aggregation": (
            "disabled"
            if mode is MultiAgentPi05Mode.INDEPENDENT
            else "independent_softmax_role_concat"
        ),
        "interaction_channels": {
            "common": mode.common,
            "private_kv": mode.private_kv,
            "residual_action": mode.residual_action,
        },
    }
    plan = {
        **shared_protocol,
        "stage": args.stage.value,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_steps": list(permanent_checkpoint_steps),
        "recovery_checkpoint_every": args.recovery_checkpoint_every,
        "checkpoint_schedule": {
            str(step): kind for step, kind in checkpoint_schedule.items()
        },
        "learning_rates": {
            "action_expert": {
                "peak": args.action_expert_peak,
                "final": args.action_expert_final,
            },
            "paligemma": {
                "peak": args.paligemma_peak,
                "final": args.paligemma_final,
            },
            "adapter": {
                "peak": args.adapter_peak,
                "final": args.adapter_final,
            },
        },
        "model_seed": args.model_seed,
        "training_seed": args.training_seed,
        "dataset": str(args.dataset.resolve()),
        "quantiles": str(args.quantiles.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "topology": {
            "agent_groups": topology.agent_groups,
            "role_data_parallel_groups": topology.role_data_parallel_groups,
        },
        "protocol_hash": protocol_hash(shared_protocol),
        "stage1_checkpoint": None
        if args.stage1_checkpoint is None
        else str(args.stage1_checkpoint.resolve()),
        "resume_checkpoint": None
        if args.resume_checkpoint is None
        else str(args.resume_checkpoint.resolve()),
        "stop_after": args.steps if args.stop_after is None else args.stop_after,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print(
            "DRY_RUN_ONLY: add --execute inside a guarded agent-rank launcher",
            file=sys.stderr,
        )
        return

    initialize_distributed(args)
    rank = jax.process_index()
    role = topology.role_for_rank(rank)
    primary = rank == 0
    if jax.local_device_count() != 1:
        raise ValueError("each agent process must see exactly one GPU")
    print(
        f"rank={rank} role={role} pid={os.getpid()} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    if primary:
        args.output.mkdir(parents=True, exist_ok=args.resume_checkpoint is not None)
        (args.output / "launch_manifest.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sync("multiarm_frame_output_ready")

    load_started = time.monotonic()
    _, model = load_local_role_pi05(
        args.base_checkpoint,
        role_index=role,
        agent_count=agent_count,
        max_agents=4,
        seed=args.model_seed,
        train_action_ffw=False,
        train_action_attention=True,
        train_paligemma_kv=True,
        action_horizon=50,
    )
    settings = GroupedOptimizerSettings(
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        action_expert_peak=args.action_expert_peak,
        action_expert_final=args.action_expert_final,
        paligemma_peak=args.paligemma_peak,
        paligemma_final=args.paligemma_final,
        adapter_peak=args.adapter_peak,
        adapter_final=args.adapter_final,
    )
    engine = create_multiagent_engine(
        model,
        mode=mode,
        topology=topology,
        settings=settings,
    )
    del model
    gc.collect()
    sync("multiarm_frame_models_loaded")
    print(f"rank={rank} model_load_seconds={time.monotonic() - load_started:.3f}", flush=True)

    role_protocol = {**shared_protocol, "role": role}
    parent_hash = None
    run_rng = jax.random.key(args.training_seed)
    checkpoint_source = args.stage1_checkpoint or args.resume_checkpoint
    if checkpoint_source is not None:
        restore_started = time.monotonic()
        print(
            f"rank={rank} checkpoint_restore_start={checkpoint_source}",
            flush=True,
        )
        snapshot = restore_role_snapshot(
            checkpoint_source,
            role=role,
            expected_agent_count=agent_count,
        )
        print(
            f"rank={rank} checkpoint_deserialize_seconds="
            f"{time.monotonic() - restore_started:.3f}",
            flush=True,
        )
        engine_restore_started = time.monotonic()
        if args.stage1_checkpoint is not None:
            parent_hash = fork_role_engine_from_stage1(
                engine,
                snapshot,
                training_seed=args.training_seed,
                expected_stage1=STAGE_PARENT[args.stage],
            )
        else:
            run_rng, parent_hash = resume_role_engine(
                engine,
                snapshot,
                expected_stage=args.stage,
            )
        del snapshot
        gc.collect()
        print(
            f"rank={rank} checkpoint_engine_restore_seconds="
            f"{time.monotonic() - engine_restore_started:.3f}",
            flush=True,
        )
    if checkpoint_source is None:
        sync("multiarm_frame_state_initialized")
    else:
        # The first training step already provides the required four-rank
        # rendezvous. A separate GPU collective immediately after restoring
        # multi-GiB device state can deadlock behind asynchronous transfers.
        print(
            f"rank={rank} checkpoint_restore_ready step={engine.current_step()}",
            flush=True,
        )

    data_started = time.monotonic()
    normalization = MultiArmQuantileNormalization.from_npz(
        args.quantiles,
        expected_agent_count=agent_count,
    )
    dataset = MultiArmTaskDataset(
        args.dataset,
        agent_count=agent_count,
        action_horizon=50,
    )
    tokenizer = PaligemmaTokenizer(200)
    checkpoint_targets = checkpoint_schedule
    samples_per_step = batch.team_microbatch * batch.accumulation

    stop_step = args.steps if args.stop_after is None else args.stop_after
    if engine.current_step() >= stop_step:
        raise ValueError("requested stop is not later than the restored step")
    first_optimizer_step = engine.current_step()
    print(
        f"rank={rank} dataset_ready_seconds={time.monotonic() - data_started:.3f} "
        f"resume_step={first_optimizer_step}",
        flush=True,
    )
    for optimizer_step in range(engine.current_step(), stop_step):
        if optimizer_step == first_optimizer_step:
            print(f"rank={rank} first_step_start={optimizer_step + 1}", flush=True)
        samples = []
        for sample_index in range(samples_per_step):
            data_rng = data_rng_for_sample(
                seed=args.training_seed,
                step=optimizer_step,
                sample_index=sample_index,
            )
            trajectory_index, episode_step = dataset.sample_location(data_rng)
            samples.append(
                build_role_training_sample(
                    dataset.team_sample(
                        trajectory_index=trajectory_index,
                        step=episode_step,
                    ).roles[role],
                    normalization=normalization,
                    tokenizer=tokenizer,
                    instruction=args.instruction,
                )
            )
        observation, actions, ownership = device_sample(
            stack_role_training_samples(samples)
        )
        step_rng = role_rng(
            seed=args.training_seed,
            step=optimizer_step,
            microstep=0,
            team_index=topology.team_for_rank(rank),
            role_index=role,
        )
        info = engine.step(
            step_rng,
            observation,
            actions,
            ownership,
            preprocessed=False,
            gradient_accumulation=batch.accumulation,
        )
        finite = bool(np.asarray(jax.device_get(info.finite)).all())
        if not finite:
            raise FloatingPointError(f"rank {rank} rejected non-finite update")
        current_step = engine.current_step()
        print(
            f"rank={rank} step={current_step} "
            f"team_loss={host_scalar(info.team_loss):.6f} "
            f"local_loss={host_scalar(info.local_loss):.6f} "
            f"role_losses={np.asarray(jax.device_get(info.role_losses)).reshape(-1).tolist()} "
            f"grad_norm={host_scalar(info.grad_norm):.6f}",
            flush=True,
        )
        if current_step in checkpoint_targets:
            checkpoint_kind = checkpoint_targets[current_step]
            checkpoint_root = (
                "checkpoints" if checkpoint_kind == "permanent" else "recovery"
            )
            checkpoint = args.output / checkpoint_root / f"step_{current_step:08d}"
            snapshot = snapshot_role_engine(
                engine,
                stage=args.stage,
                rng=run_rng,
                parent_model_sha256=parent_hash,
                model_seed=args.model_seed,
                training_seed=args.training_seed,
                protocol_metadata=role_protocol,
            )
            save_role_checkpoint(checkpoint, role=role, snapshot=snapshot)
            del snapshot
            gc.collect()
            sync(f"multiarm_frame_checkpoint_shards_{current_step}")
            if primary:
                finalize_team_checkpoint(
                    checkpoint,
                    agent_count=agent_count,
                    protocol_hash=plan["protocol_hash"],
                )
            sync(f"multiarm_frame_checkpoint_complete_{current_step}")
            if primary:
                recovery_root = args.output / "recovery"
                if recovery_root.exists():
                    for old_checkpoint in recovery_root.glob("step_*"):
                        if old_checkpoint != checkpoint:
                            shutil.rmtree(old_checkpoint)
            sync(f"multiarm_frame_recovery_cleanup_{current_step}")
            print(
                f"rank={rank} CHECKPOINT={checkpoint} kind={checkpoint_kind}",
                flush=True,
            )

    digest = parameter_sha256(engine.selected_params().to_pure_dict())
    print(f"rank={rank} final_step={engine.current_step()} model_sha256={digest}", flush=True)
    memory = jax.local_devices()[0].memory_stats() or {}
    print(
        "JAX_MEMORY_STATS "
        + json.dumps(
            {
                key: int(value)
                for key, value in memory.items()
                if isinstance(value, (int, np.integer))
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dataset.close()
    sync("multiarm_frame_training_finished")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
