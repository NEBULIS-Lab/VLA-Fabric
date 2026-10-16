#!/usr/bin/env python
"""Run fixed-condition RoboTwin2 rollouts against a local PI0.5 server."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from pi05_fabric.evaluation.robotwin2_adapter import observation_to_policy_payload
from pi05_fabric.evaluation.robotwin2_adapter import policy_r6_chunk_to_robotwin_ee
from pi05_fabric.evaluation.robotwin2_ipc import receive_message
from pi05_fabric.evaluation.robotwin2_ipc import send_message
from pi05_fabric.evaluation.robotwin2_ipc import validate_policy_response
from pi05_fabric.evaluation.robotwin2_protocol import audit_robotwin_rollout_rows
from pi05_fabric.evaluation.robotwin2_protocol import load_condition_csv
from pi05_fabric.evaluation.robotwin2_protocol import summarize_phase_metrics
from pi05_fabric.evaluation.robotwin2_task_diagnostics import PHASE_FIELDS
from pi05_fabric.evaluation.robotwin2_task_diagnostics import task_diagnostics


_FIELDS = (
    "condition_id",
    "env_seed",
    "instruction",
    "success",
    "steps",
    "planning_calls",
    "mean_plan_seconds",
    "p95_plan_seconds",
    "episode_seconds",
    "block_lifted",
    "middle_reached",
    "target_region_reached",
    "timeout",
    "error",
    "task",
    "phase_metrics_status",
    "official_step_limit",
    "effective_step_limit",
    "termination_reason",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--conditions-file", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "fresh"), required=True)
    parser.add_argument("--condition-ids")
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default="handover_block")
    parser.add_argument("--task-config", default="demo_clean_novideo")
    parser.add_argument("--prediction-horizon", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=25)
    parser.add_argument("--maximum-steps", type=int, default=800)
    parser.add_argument("--max-rollout-seconds", type=float, default=1800.0)
    parser.add_argument("--endpoint-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--purpose",
        choices=(
            "readiness_smoke",
            "checkpoint_screen",
            "development_validation",
            "fresh_test",
            "no_retraining_diagnostic",
        ),
        required=True,
    )
    return parser.parse_args()


def _task_instance(task_name: str):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def _embodiment_config(robot_file: str) -> dict:
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as stream:
        return yaml.load(stream.read(), Loader=yaml.FullLoader)


def _prepare_env_args(root: Path, task_name: str, task_config: str) -> dict:
    from envs import CONFIGS_PATH

    with (root / "task_config" / f"{task_config}.yml").open("r", encoding="utf-8") as stream:
        args = yaml.load(stream.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as stream:
        embodiment_types = yaml.load(stream.read(), Loader=yaml.FullLoader)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as stream:
        camera_types = yaml.load(stream.read(), Loader=yaml.FullLoader)
    robot_file = embodiment_types[args["embodiment"][0]]["file_path"]
    args.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "ckpt_setting": "demo_clean",
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "dual_arm_embodied": True,
            "left_embodiment_config": _embodiment_config(robot_file),
            "right_embodiment_config": _embodiment_config(robot_file),
            "head_camera_h": camera_types[args["camera"]["head_camera_type"]]["h"],
            "head_camera_w": camera_types[args["camera"]["head_camera_type"]]["w"],
            "policy_name": "pi05_fabric_bridge",
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": False,
            "collect_data": False,
        }
    )
    return args


def _wait_endpoint(path: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"policy endpoint was not published within {timeout:.1f}s")
        time.sleep(0.1)
    endpoint = json.loads(path.read_text(encoding="utf-8"))
    if endpoint.get("status") != "READY":
        raise RuntimeError(f"policy endpoint is not ready: {endpoint}")
    return endpoint


def _close(environment) -> None:
    try:
        environment.close_env(clear_cache=True)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)


def _phase_state(environment, *, initial_z: float) -> tuple[int, int, int]:
    return task_diagnostics("handover_block").observe(environment, initial_z)


def _failure_row(
    condition, *, started: float, timeout: bool, error: str,
    task_name: str = "handover_block",
) -> dict:
    diagnostics = task_diagnostics(task_name)
    return {
        "condition_id": condition.condition_id,
        "env_seed": condition.env_seed,
        "instruction": condition.instruction,
        "success": 0,
        "steps": 0,
        "planning_calls": 0,
        "mean_plan_seconds": 0.0,
        "p95_plan_seconds": 0.0,
        "episode_seconds": time.monotonic() - started,
        **dict(zip(PHASE_FIELDS, diagnostics.initial_phases)),
        "timeout": int(timeout),
        "error": error,
        "task": diagnostics.task,
        "phase_metrics_status": diagnostics.phase_metrics_status,
        "official_step_limit": None,
        "effective_step_limit": None,
        "termination_reason": "wall_timeout" if timeout else "error",
    }


def _run_condition(connection: socket.socket, condition, env_args: dict, args) -> dict:
    started = time.monotonic()
    np.random.seed(condition.env_seed)
    torch.manual_seed(condition.env_seed)
    diagnostics = task_diagnostics(args.task_name)
    environment = _task_instance(args.task_name)
    plan_times = []
    planning_calls = 0
    phases = diagnostics.initial_phases
    official_step_limit = effective_step_limit = None
    try:
        environment.setup_demo(
            now_ep_num=condition.condition_id,
            seed=condition.env_seed,
            is_test=True,
            **env_args,
        )
        official_step_limit = int(environment.step_lim)
        cap = min(args.maximum_steps, 800) if diagnostics.task == "robotwin_handover_mic" else args.maximum_steps
        effective_step_limit = min(official_step_limit, cap)
        if effective_step_limit <= 0:
            raise ValueError("effective step limit must be positive")
        environment.step_lim = effective_step_limit
        environment.set_instruction(instruction=condition.instruction)
        initial_state = diagnostics.start(environment)
        while environment.take_action_cnt < environment.step_lim and not environment.eval_success:
            if time.monotonic() - started > args.max_rollout_seconds:
                raise TimeoutError("rollout exceeded wall-clock limit")
            observation = environment.get_obs()
            request_id = f"{condition.condition_id}:{planning_calls}"
            request = observation_to_policy_payload(
                observation,
                instruction=condition.instruction,
                condition_id=condition.condition_id,
                planning_call=planning_calls,
                request_id=request_id,
            )
            call_started = time.monotonic()
            send_message(connection, request)
            actions_r6 = validate_policy_response(
                receive_message(connection), request_id=request_id
            )
            plan_times.append(time.monotonic() - call_started)
            actions_ee = policy_r6_chunk_to_robotwin_ee(actions_r6)
            planning_calls += 1
            for action in actions_ee[: args.execution_horizon]:
                if environment.take_action_cnt >= environment.step_lim or environment.eval_success:
                    break
                if time.monotonic() - started > args.max_rollout_seconds:
                    raise TimeoutError("rollout exceeded wall-clock limit")
                environment.take_action(action, action_type="ee")
                phase = diagnostics.observe(environment, initial_state)
                phases = tuple(
                    None if current is None else max(previous, current)
                    for previous, current in zip(phases, phase)
                )
        success = int(bool(environment.eval_success or environment.check_success()))
        return {
            "condition_id": condition.condition_id,
            "env_seed": condition.env_seed,
            "instruction": condition.instruction,
            "success": success,
            "steps": int(environment.take_action_cnt),
            "planning_calls": planning_calls,
            "mean_plan_seconds": float(np.mean(plan_times)) if plan_times else 0.0,
            "p95_plan_seconds": float(np.percentile(plan_times, 95)) if plan_times else 0.0,
            "episode_seconds": time.monotonic() - started,
            **dict(zip(PHASE_FIELDS, phases)),
            "timeout": 0,
            "error": "",
            "task": diagnostics.task,
            "phase_metrics_status": diagnostics.phase_metrics_status,
            "official_step_limit": official_step_limit,
            "effective_step_limit": effective_step_limit,
            "termination_reason": "success" if success else "step_limit",
        }
    except Exception as exc:
        traceback.print_exc()
        row = _failure_row(
            condition, started=started, timeout=isinstance(exc, TimeoutError),
            error=f"{type(exc).__name__}: {exc}", task_name=args.task_name,
        )
        row.update({
            "steps": int(getattr(environment, "take_action_cnt", 0)),
            "planning_calls": planning_calls,
            "mean_plan_seconds": float(np.mean(plan_times)) if plan_times else 0.0,
            "p95_plan_seconds": float(np.percentile(plan_times, 95)) if plan_times else 0.0,
            "official_step_limit": official_step_limit,
            "effective_step_limit": effective_step_limit,
            **dict(zip(PHASE_FIELDS, phases)),
        })
        return row
    finally:
        _close(environment)


def main() -> None:
    args = _parse_args()
    diagnostics = task_diagnostics(args.task_name)
    args.task_name = diagnostics.task.removeprefix("robotwin_")
    if args.prediction_horizon != 50:
        raise ValueError("PI0.5 V2 evaluation requires prediction horizon 50")
    if not 1 <= args.execution_horizon <= 50:
        raise ValueError("execution horizon must be between 1 and 50")
    if args.maximum_steps <= 0:
        raise ValueError("maximum steps must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    expected_ids = tuple(range(200)) if args.split == "validation" else tuple(range(200, 400))
    all_conditions = load_condition_csv(args.conditions_file, expected_ids=expected_ids)
    if args.condition_ids:
        selected_ids = tuple(int(item) for item in args.condition_ids.split(","))
        selected = tuple(item for item in all_conditions if item.condition_id in selected_ids)
        if tuple(item.condition_id for item in selected) != selected_ids:
            raise ValueError("requested condition IDs are missing or out of order")
    else:
        selected = all_conditions
    root = args.robotwin_root.resolve()
    for path in (root, root / "script", root / "policy", root / "description" / "utils"):
        sys.path.insert(0, str(path))
    os.chdir(root)
    env_args = _prepare_env_args(root, args.task_name, args.task_config)
    endpoint = _wait_endpoint(args.endpoint_file, args.endpoint_timeout)
    endpoint_task = endpoint.get('task')
    task_mismatch = (
        endpoint_task != diagnostics.task
        if diagnostics.task != "robotwin_handover_block"
        else endpoint_task not in (None, diagnostics.task)
    )
    if task_mismatch:
        if diagnostics.task == "robotwin_handover_mic":
            raise ValueError("Mic evaluator requires a Mic policy endpoint")
        raise ValueError(
            f"{diagnostics.task} evaluator requires a matching policy endpoint, "
            f"got {endpoint.get('task')!r}"
        )
    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "task": diagnostics.task,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "robotwin_root": str(root),
        "phase_metrics_status": diagnostics.phase_metrics_status,
        "purpose": args.purpose,
        "split": args.split,
        "condition_ids": [item.condition_id for item in selected],
        "prediction_horizon": args.prediction_horizon,
        "execution_horizon": args.execution_horizon,
        "maximum_steps": args.maximum_steps,
        "effective_step_limit_rule": (
            "min(official_step_limit, maximum_steps, 800)"
            if diagnostics.task == "robotwin_handover_mic"
            else "min(official_step_limit, maximum_steps)"
        ),
        "official_success_unchanged": True,
        "pair_environment_seed": True,
        "pair_initial_state": True,
        "pair_flow_noise": True,
        "endpoint": endpoint,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    partial = args.output_dir / "rollouts.partial.csv"
    with socket.create_connection((endpoint["host"], endpoint["port"]), timeout=1200) as connection:
        connection.settimeout(1200)
        with partial.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_FIELDS)
            writer.writeheader()
            for index, condition in enumerate(selected, start=1):
                started = time.monotonic()
                try:
                    row = _run_condition(connection, condition, env_args, args)
                except Exception as exc:
                    traceback.print_exc()
                    row = _failure_row(
                        condition,
                        started=started,
                        timeout=isinstance(exc, TimeoutError),
                        error=f"{type(exc).__name__}: {exc}",
                        task_name=args.task_name,
                    )
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                print(f"ROBOTWIN_ROLLOUT {index}/{len(selected)} {json.dumps(row)}", flush=True)
    audit_robotwin_rollout_rows(rows, conditions=selected)
    summary = {
        "schema_version": 1,
        "status": "PASS" if not any(row["error"] for row in rows) else "FAIL",
        "purpose": args.purpose,
        "successes": sum(int(row["success"]) for row in rows),
        "trials": len(rows),
        "timeouts": sum(int(row["timeout"]) for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "mean_planning_calls": float(np.mean([row["planning_calls"] for row in rows])),
        **summarize_phase_metrics(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.rename(args.output_dir / "rollouts.csv")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise RuntimeError("one or more RoboTwin rollouts failed")


if __name__ == "__main__":
    main()
