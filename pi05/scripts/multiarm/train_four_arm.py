#!/usr/bin/env python3
"""Train one four-agent MuJoCo stage; dry-run unless --execute is set."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
from pi05_fabric.data.four_arm_tasks import FourArmQuantileNormalization
from pi05_fabric.data.four_arm_tasks import FourArmTaskDataset
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
from pi05_fabric.training.protocol import checkpoint_steps
from pi05_fabric.training.protocol import data_rng_for_sample
from pi05_fabric.training.stages import StageName
from pi05_fabric.training.stages import parameter_sha256


STAGES = (
    StageName.PI05_FOUR_ARM_COMMON_STAGE1,
    StageName.PI05_FOUR_ARM_FULL_STAGE2,
)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=StageName, choices=STAGES, required=True)
    parser.add_argument("--task", choices=("frame_insertion", "arch_assembly"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quantiles", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=root / "external/pi05_base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--team-microbatch", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--model-seed", type=int, default=731)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--distributed-process-count", type=int, default=4)
    parser.add_argument("--distributed-process-id", type=int, default=0)
    parser.add_argument("--distributed-coordinator-address")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--allow-nonformal-steps", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def initialize_distributed(args) -> bool:
    if args.distributed_process_count == 1:
        raise ValueError("four-arm agent parallelism requires four processes")
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


def recorded_file_sha256(path: Path, manifest: Path) -> str:
    target = path.resolve()
    manifest_root = manifest.resolve().parent
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        candidate = (manifest_root / relative).resolve()
        if candidate == target:
            if len(digest) != 64:
                raise ValueError(f"invalid SHA-256 record for {path}")
            return digest
    raise ValueError(f"no audited SHA-256 record for {path}")


def validate_args(args, topology):
    if topology.world_size != 4:
        raise ValueError("the formal four-arm run requires exactly four ranks")
    if (
        not args.dataset.is_file()
        or not args.quantiles.is_file()
        or not args.audit_manifest.is_file()
    ):
        raise FileNotFoundError("audited four-arm data or quantiles are missing")
    if not args.base_checkpoint.exists():
        raise FileNotFoundError("PI0.5 base checkpoint is missing")
    if args.steps <= 0 or args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("invalid step or warmup configuration")
    expected_steps = {
        StageName.PI05_FOUR_ARM_COMMON_STAGE1: 10_000,
        StageName.PI05_FOUR_ARM_FULL_STAGE2: 50_000,
    }
    if not args.allow_nonformal_steps and args.steps != expected_steps[args.stage]:
        raise ValueError("formal Stage 1/2 lengths are fixed at 10k/50k")
    if args.stage is StageName.PI05_FOUR_ARM_COMMON_STAGE1:
        if args.stage1_checkpoint is not None:
            raise ValueError("Stage 1 cannot load a Stage 1 parent")
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
    topology = AgentParallelTopology.create(
        world_size=args.distributed_process_count,
        agent_count=4,
    )
    batch = validate_args(args, topology)
    mode = (
        MultiAgentPi05Mode.COMMON_ONLY
        if args.stage is StageName.PI05_FOUR_ARM_COMMON_STAGE1
        else MultiAgentPi05Mode.FULL
    )
    shared_protocol = {
        "schema_version": 1,
        "task": args.task,
        "instruction": args.instruction,
        "agent_count": 4,
        "mode": mode.value,
        "action_horizon": 50,
        "execution_horizon": 25,
        "flow_steps": 10,
        "flow_noise": "role_local_deterministic",
        "normalization": "q01_q99_per_role",
        "dataset_sha256": recorded_file_sha256(args.dataset, args.audit_manifest),
        "normalization_sha256": recorded_file_sha256(args.quantiles, args.audit_manifest),
        "global_team_batch": batch.global_team_batch,
        "team_microbatch": batch.team_microbatch,
        "gradient_accumulation": batch.accumulation,
        "train_action_expert_ffw_base": False,
        "train_action_expert_attention": True,
        "train_paligemma_kv": True,
        "remote_action_aggregation": "independent_softmax_role_concat",
    }
    plan = {
        **shared_protocol,
        "stage": args.stage.value,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_steps": list(
            checkpoint_steps(total_steps=args.steps, every=args.checkpoint_every)
        ),
        "model_seed": args.model_seed,
        "training_seed": args.training_seed,
        "dataset": str(args.dataset.resolve()),
        "quantiles": str(args.quantiles.resolve()),
        "audit_manifest": str(args.audit_manifest.resolve()),
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
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print(
            "DRY_RUN_ONLY: add --execute inside a guarded four-rank launcher",
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
    sync("four_arm_output_ready")

    load_started = time.monotonic()
    _, model = load_local_role_pi05(
        args.base_checkpoint,
        role_index=role,
        agent_count=4,
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
        action_expert_peak=5e-6,
        action_expert_final=5e-7,
        paligemma_peak=3e-6,
        paligemma_final=3e-7,
        adapter_peak=5e-5,
        adapter_final=5e-6,
    )
    engine = create_multiagent_engine(
        model,
        mode=mode,
        topology=topology,
        settings=settings,
    )
    del model
    gc.collect()
    sync("four_arm_models_loaded")
    print(f"rank={rank} model_load_seconds={time.monotonic() - load_started:.3f}", flush=True)

    parent_hash = None
    run_rng = jax.random.key(args.training_seed)
    if args.stage1_checkpoint is not None:
        stage1 = restore_role_snapshot(
            args.stage1_checkpoint,
            role=role,
            expected_agent_count=4,
        )
        parent_hash = fork_role_engine_from_stage1(
            engine,
            stage1,
            training_seed=args.training_seed,
            expected_stage1=StageName.PI05_FOUR_ARM_COMMON_STAGE1,
        )
        del stage1
        gc.collect()
    elif args.resume_checkpoint is not None:
        resumed = restore_role_snapshot(
            args.resume_checkpoint,
            role=role,
            expected_agent_count=4,
        )
        run_rng, parent_hash = resume_role_engine(
            engine,
            resumed,
            expected_stage=args.stage,
        )
        del resumed
        gc.collect()
    sync("four_arm_state_initialized")

    normalization = FourArmQuantileNormalization.from_npz(args.quantiles)
    dataset = FourArmTaskDataset(args.dataset, action_horizon=50)
    tokenizer = PaligemmaTokenizer(200)
    checkpoint_targets = set(
        checkpoint_steps(total_steps=args.steps, every=args.checkpoint_every)
    )
    samples_per_step = batch.team_microbatch * batch.accumulation
    role_protocol = {**shared_protocol, "role": role}

    for optimizer_step in range(engine.current_step(), args.steps):
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
            checkpoint = args.output / "checkpoints" / f"step_{current_step:08d}"
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
            sync(f"four_arm_checkpoint_shards_{current_step}")
            if primary:
                finalize_team_checkpoint(
                    checkpoint,
                    agent_count=4,
                    protocol_hash=plan["protocol_hash"],
                )
            sync(f"four_arm_checkpoint_complete_{current_step}")
            print(f"rank={rank} CHECKPOINT={checkpoint}", flush=True)

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
    sync("four_arm_training_finished")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
