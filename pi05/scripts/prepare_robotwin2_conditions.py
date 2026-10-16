#!/usr/bin/env python
"""Freeze existing or generate expert-validated RoboTwin2 conditions."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import yaml

from pi05_fabric.evaluation.robotwin2_protocol import RobotwinCondition
from pi05_fabric.evaluation.robotwin2_protocol import freeze_condition_split
from pi05_fabric.evaluation.robotwin2_protocol import merge_condition_csvs
from pi05_fabric.evaluation.robotwin2_protocol import write_condition_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--split", choices=("validation", "fresh"), required=True)
    freeze.add_argument("--first-id", type=int, required=True)
    freeze.add_argument("--count", type=int, default=200)
    freeze.add_argument("--expert-validated", action="store_true")
    freeze.add_argument("--task-name", default="robotwin_handover_block")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--robotwin-root", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--first-id", type=int, required=True)
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument("--seed-start", type=int, required=True)
    generate.add_argument("--task-name", default="handover_block")
    generate.add_argument("--task-config", default="demo_clean_novideo")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--sources", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--first-id", type=int, required=True)
    merge.add_argument("--count", type=int, required=True)
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


def _close(environment) -> None:
    try:
        environment.close_env(clear_cache=True)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)


def _generate(args: argparse.Namespace) -> None:
    root = args.robotwin_root.resolve()
    for path in (root, root / "script", root / "policy", root / "description" / "utils"):
        sys.path.insert(0, str(path))
    os.chdir(root)
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    env_args = _prepare_env_args(root, args.task_name, args.task_config)
    conditions: list[RobotwinCondition] = []
    seed = args.seed_start
    while len(conditions) < args.count:
        np.random.seed(seed)
        torch.manual_seed(seed)
        environment = _task_instance(args.task_name)
        try:
            environment.setup_demo(
                now_ep_num=args.first_id + len(conditions), seed=seed, is_test=True, **env_args
            )
            episode_info = environment.play_once()
            valid = bool(environment.plan_success and environment.check_success())
            if valid:
                descriptions = generate_episode_descriptions(
                    args.task_name, [episode_info["info"]], 1
                )[0]["unseen"]
                choices = tuple(sorted({str(item).strip() for item in descriptions if str(item).strip()}))
                if not choices:
                    raise RuntimeError("expert episode produced no non-empty unseen instruction")
                condition_id = args.first_id + len(conditions)
                instruction = choices[condition_id % len(choices)]
                conditions.append(RobotwinCondition(condition_id, seed, instruction))
                print(
                    f"EXPERT_VALID {len(conditions)}/{args.count} id={condition_id} seed={seed}",
                    flush=True,
                )
        except UnStableError:
            pass
        finally:
            _close(environment)
        seed += 1
    write_condition_csv(args.output, conditions)
    print(json.dumps({"status": "PASS", "count": len(conditions), "next_seed": seed}))


def main() -> None:
    args = _parse_args()
    if args.command == "freeze":
        expected = tuple(range(args.first_id, args.first_id + args.count))
        path = freeze_condition_split(
            args.source,
            output_dir=args.output_dir,
            split=args.split,
            expected_ids=expected,
            expert_validated=args.expert_validated,
            task_name=args.task_name,
        )
        print(path)
    elif args.command == "generate":
        _generate(args)
    else:
        merged = merge_condition_csvs(
            args.sources,
            output=args.output,
            expected_ids=tuple(range(args.first_id, args.first_id + args.count)),
        )
        print(json.dumps({"status": "PASS", "count": len(merged), "output": str(args.output)}))


if __name__ == "__main__":
    main()
