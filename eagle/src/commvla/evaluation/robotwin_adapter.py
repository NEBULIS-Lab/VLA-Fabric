"""Small compatibility layer for launching RoboTwin environments."""

from __future__ import annotations

import importlib
import os
import traceback
from contextlib import contextmanager
from pathlib import Path

import yaml


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(envs_module, task_name)()
    except Exception as exc:
        raise RuntimeError(f"Unknown RoboTwin task: {task_name}") from exc


def _embodiment_config(robot_file: str) -> dict:
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as handle:
        return yaml.load(handle.read(), Loader=yaml.FullLoader)


def prepare_robotwin_args(task_name: str, task_config: str, ckpt_setting: str, policy_name: str, seed: int):
    from envs import CONFIGS_PATH

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)

    args.update(task_name=task_name, task_config=task_config, ckpt_setting=ckpt_setting)
    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    def embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"Missing embodiment file for {name}")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as handle:
        camera_config = yaml.load(handle.read(), Loader=yaml.FullLoader)
    camera = camera_config[args["camera"]["head_camera_type"]]
    args["head_camera_h"] = camera["h"]
    args["head_camera_w"] = camera["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("RoboTwin embodiment must contain one or three entries")

    args["left_embodiment_config"] = _embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _embodiment_config(args["right_robot_file"])
    args["policy_name"] = policy_name
    args["eval_mode"] = True
    user_args = {
        "task_name": task_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "policy_name": policy_name,
        "instruction_type": "unseen",
        "seed": seed,
        "left_arm_dim": len(args["left_embodiment_config"]["arm_joints_name"][0]),
        "right_arm_dim": len(args["right_embodiment_config"]["arm_joints_name"][1]),
    }
    return args, user_args


def close_env_safely(task_env, clear_cache: bool = False) -> None:
    if task_env is None:
        return
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        print("[commvla] RoboTwin environment cleanup failed", flush=True)
        print(traceback.format_exc(), flush=True)


def find_valid_episode(task_name: str, args: dict, episode_id: int, seed: int, max_attempts: int = 500):
    """Find a RoboTwin initialization for which the scripted expert succeeds."""
    for _ in range(max_attempts):
        task_env = class_decorator(task_name)
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **args)
            episode_info = task_env.play_once()
            valid = bool(task_env.plan_success and task_env.check_success())
        except Exception:
            close_env_safely(task_env)
            seed += 1
            continue
        close_env_safely(task_env)
        if valid:
            return seed, episode_info
        seed += 1
    raise RuntimeError(f"No valid expert episode for {task_name} in {max_attempts} attempts")
