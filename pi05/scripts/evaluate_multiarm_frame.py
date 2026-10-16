#!/usr/bin/env python3
"""Run strict agent-parallel PI0.5 rollouts for multi-arm frame tasks."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import time
import traceback

import jax
from jax.experimental import multihost_utils
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer

from pi05_fabric.agents.load_pi05 import load_local_role_pi05
from pi05_fabric.agents.multiagent_pi05 import MultiAgentPi05Mode
from pi05_fabric.data.multiarm_frame_tasks import MultiArmQuantileNormalization
from pi05_fabric.data.four_arm_pi05 import build_role_policy_observation
from pi05_fabric.evaluation.multiagent_inference import MultiAgentInferenceEngine
from pi05_fabric.evaluation.four_arm_ipc import receive_pickle
from pi05_fabric.evaluation.four_arm_ipc import send_pickle
from pi05_fabric.evaluation.four_arm_ipc import short_environment_socket_path
from pi05_fabric.evaluation.multiarm_frame import dispatch_role_actions

from pi05_fabric.training.agent_parallel import AgentParallelTopology
from pi05_fabric.training.multiagent_checkpoint import audit_team_checkpoint_metadata
from pi05_fabric.training.multiagent_checkpoint import load_role_weights
from pi05_fabric.training.stages import StageName


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument(
        "--task",
        choices=("frame4_insertion", "frame3_triangle_insertion"),
        required=True,
    )
    parser.add_argument("--agent-count", type=int, choices=(3, 4), required=True)
    parser.add_argument("--quantiles", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--robosuite-root", type=Path, required=True)
    parser.add_argument(
        "--environment-server",
        type=Path,
        default=root / "scripts/multiarm_frame_env_server.py",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--maximum-steps", type=int, default=800)
    parser.add_argument("--distributed-process-count", type=int, required=True)
    parser.add_argument("--distributed-process-id", type=int, default=0)
    parser.add_argument("--distributed-coordinator-address")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evaluation_plan(args: argparse.Namespace) -> dict:
    if args.seed_count <= 0:
        raise ValueError("seed_count must be positive")
    return {
        "schema_version": 1,
        "task": args.task,
        "instruction": args.instruction,
        "agent_count": args.agent_count,
        "prediction_horizon": 50,
        "execution_horizon": 25,
        "flow_steps": 10,
        "flow_noise": "role_local_deterministic",
        "termination_policy": "environment_task_status",
        "maximum_steps": args.maximum_steps,
        "seeds": list(range(args.seed_start, args.seed_start + args.seed_count)),
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
    }


def sync(name: str) -> None:
    multihost_utils.sync_global_devices(name)


def atomic_pickle(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def to_json_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_native(item) for item in value]
    return value


def connect_environment(socket_path: Path, *, timeout: float = 180.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(socket_path))
            return connection
        except OSError:
            connection.close()
            time.sleep(0.1)
    raise TimeoutError("multi-arm environment server did not accept a connection")


def launch_environment(args, socket_path: Path, log_path: Path):
    command = [
        str(args.environment_python),
        str(args.environment_server),
        "--socket", str(socket_path),
        "--task", args.task,
        "--agent-count", str(args.agent_count),
        "--maximum-steps", str(args.maximum_steps),
        "--image-size", "224",
    ]
    python_path = f"{args.robosuite_root}:{args.environment_root / 'src'}"
    environment = {
        **os.environ,
        "PYTHONPATH": python_path,
        "OMP_NUM_THREADS": "4",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
    }
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=args.environment_root, env=environment,
        stdout=log, stderr=subprocess.STDOUT,
    )
    return process, log


def initialize_distributed(args) -> None:
    if (
        args.distributed_process_count != args.agent_count
        or args.distributed_coordinator_address is None
    ):
        raise ValueError("strict multi-arm evaluation requires one rank per agent")
    jax.distributed.initialize(
        coordinator_address=args.distributed_coordinator_address,
        num_processes=args.agent_count,
        process_id=args.distributed_process_id,
        local_device_ids=0,
    )


def mode_for_stage(stage: str) -> MultiAgentPi05Mode:
    if stage == StageName.PI05_FRAME3_COMMON_STAGE1.value:
        return MultiAgentPi05Mode.COMMON_ONLY
    if stage in (
        StageName.PI05_FRAME3_FULL_STAGE2.value,
        StageName.PI05_FRAME3_FULL_DIRECT.value,
    ):
        return MultiAgentPi05Mode.FULL
    if stage in (
        StageName.PI05_FRAME4_INDEPENDENT_DIRECT.value,
        StageName.PI05_FRAME4_INDEPENDENT_STAGE1.value,
        StageName.PI05_FRAME4_INDEPENDENT_STAGE2.value,
        StageName.PI05_FRAME3_INDEPENDENT_DIRECT.value,
    ):
        return MultiAgentPi05Mode.INDEPENDENT
    raise ValueError(f"unsupported multi-arm checkpoint stage: {stage}")


def main() -> None:
    args = parse_args()
    expected = 4 if args.task == "frame4_insertion" else 3
    if args.agent_count != expected:
        raise ValueError("task and agent count are inconsistent")
    plan = evaluation_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    for path in (
        args.checkpoint,
        args.quantiles,
        args.base_checkpoint,
        args.environment_python,
        args.environment_root,
        args.robosuite_root,
        args.environment_server,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    initialize_distributed(args)
    rank = jax.process_index()
    role = rank
    primary = rank == 0
    if jax.local_device_count() != 1:
        raise ValueError("each evaluation rank must expose exactly one GPU")

    checkpoint_manifest = audit_team_checkpoint_metadata(
        args.checkpoint,
        expected_agent_count=args.agent_count,
    )
    mode = mode_for_stage(checkpoint_manifest["stage"])
    _, model = load_local_role_pi05(
        args.base_checkpoint,
        role_index=role,
        agent_count=args.agent_count,
        max_agents=4,
        seed=731,
        train_action_ffw=False,
        train_action_attention=True,
        train_paligemma_kv=True,
        action_horizon=50,
    )
    engine = MultiAgentInferenceEngine(
        model,
        checkpoint_params=load_role_weights(
            args.checkpoint,
            role=role,
            expected_agent_count=args.agent_count,
        ),
        mode=mode,
        topology=AgentParallelTopology.create(
            world_size=args.agent_count,
            agent_count=args.agent_count,
        ),
    )
    del model
    normalization = MultiArmQuantileNormalization.from_npz(
        args.quantiles,
        expected_agent_count=args.agent_count,
    )
    tokenizer = PaligemmaTokenizer(200)

    if primary:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        (args.output / "evaluation_manifest.json").write_text(
            json.dumps({**plan, "checkpoint_manifest": checkpoint_manifest}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output / "shared").mkdir()
    sync("multiarm_frame_eval_output_ready")
    shared = args.output / "shared"
    observation_path = shared / "observation.pkl"
    response_path = shared / "response.pkl"
    socket_path = short_environment_socket_path(args.output)

    connection = None
    environment_process = None
    environment_log = None
    ready = None
    records = []
    completed_successfully = False
    try:
        if primary:
            environment_process, environment_log = launch_environment(
                args,
                socket_path,
                args.output / "environment.log",
            )
            connection = connect_environment(socket_path)
            ready = receive_pickle(connection)
            if ready.get("type") != "ready":
                raise RuntimeError(f"invalid environment ready message: {ready}")
        sync("multiarm_frame_environment_ready")

        for condition_index, seed in enumerate(plan["seeds"]):
            if primary:
                print(
                    f"condition_start={condition_index + 1}/{len(plan['seeds'])} "
                    f"seed={seed}",
                    flush=True,
                )
                send_pickle(connection, {"command": "reset", "seed": seed})
                response = receive_pickle(connection)
                atomic_pickle(observation_path, response)
            sync(f"multiarm_frame_reset_{condition_index}")
            initial = read_pickle(observation_path)
            planning_round = 0
            final_response = initial
            while True:
                payload = {
                    "global_rgb": final_response["observation"]["global_rgb"],
                    "wrist_rgb": final_response["observation"]["wrist_rgb"][role],
                    "state": final_response["observation"]["qpos"][role],
                }
                observation, ownership = build_role_policy_observation(
                    role_index=role,
                    global_rgb=payload["global_rgb"],
                    wrist_rgb=payload["wrist_rgb"],
                    state=payload["state"],
                    normalization=normalization,
                    tokenizer=tokenizer,
                    instruction=args.instruction,
                )
                key = jax.random.fold_in(jax.random.key(seed), planning_round)
                key = jax.random.fold_in(key, role)
                normalized = engine.sample_actions(
                    key,
                    jax.tree.map(jax.numpy.asarray, observation),
                    jax.numpy.asarray(ownership),
                    num_steps=10,
                )
                local_actions = normalization.unnormalize_actions(
                    role,
                    np.asarray(jax.device_get(normalized)),
                )
                gathered = np.asarray(
                    multihost_utils.process_allgather(local_actions, tiled=False)
                )
                if gathered.shape != (args.agent_count, 1, 50, 7):
                    raise ValueError(f"unexpected gathered action shape: {gathered.shape}")
                planning_round += 1
                if primary:
                    dispatched = dispatch_role_actions(
                        tuple(
                            gathered[index, 0]
                            for index in range(args.agent_count)
                        ),
                        agent_count=args.agent_count,
                        execution_horizon=25,
                    )
                    actions = np.stack(dispatched)
                    actions = np.clip(actions, -1.0, 1.0)
                    send_pickle(connection, {"command": "step_chunk", "actions": actions})
                    final_response = receive_pickle(connection)
                    atomic_pickle(response_path, final_response)
                sync(f"multiarm_frame_step_{condition_index}_{planning_round}")
                final_response = read_pickle(response_path)
                done = bool(
                    final_response.get("success", False)
                    or final_response.get("truncated", False)
                )
                if done:
                    break
            if primary:
                record = {
                    "seed": seed,
                    "planning_rounds": planning_round,
                    "elapsed_steps": int(final_response.get("elapsed_steps", 0)),
                    "success": bool(final_response.get("success", False)),
                    "task_status": final_response.get("task_status", {}),
                    "initial_condition": initial.get("initial_condition", {}),
                }
                record = to_json_native(record)
                records.append(record)
                with (args.output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"condition_complete={condition_index + 1}/{len(plan['seeds'])} "
                    f"seed={seed} success={record['success']} "
                    f"elapsed_steps={record['elapsed_steps']}",
                    flush=True,
                )
            sync(f"multiarm_frame_condition_complete_{condition_index}")

        if primary:
            summary = {
                "status": "COMPLETE",
                "trials": len(records),
                "successes": sum(record["success"] for record in records),
                "seeds": plan["seeds"],
                "finite_actions": True,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        sync("multiarm_frame_evaluation_complete")
        completed_successfully = True
    except BaseException:
        print(f"rank={rank} EVALUATION_RANK_FAILURE", flush=True)
        traceback.print_exc()
        raise
    finally:
        if primary and connection is not None:
            try:
                send_pickle(connection, {"command": "close"})
                receive_pickle(connection)
            except Exception:
                if completed_successfully:
                    raise
            finally:
                connection.close()
        if primary and environment_process is not None:
            environment_process.terminate()
            environment_process.wait(timeout=30)
        if environment_log is not None:
            environment_log.close()
        if completed_successfully:
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
