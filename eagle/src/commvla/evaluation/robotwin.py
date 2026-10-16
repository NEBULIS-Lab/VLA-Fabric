"""Paired RoboTwin2 rollout evaluator for official TwinVLA and CommVLA-native."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


class RolloutTimeout(RuntimeError):
    pass


@contextmanager
def rollout_timer(seconds: float):
    if seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame):
        raise RolloutTimeout(f"rollout exceeded {seconds:.1f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def add_robotwin_paths(root: Path) -> None:
    paths = [
        root,
        root / "script",
        root / "policy",
        root / "description" / "utils",
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_conditions(path: Path, offset: int, trials: int) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["rollout_id"]) >= offset]
    if trials > 0:
        selected = selected[:trials]
    if not selected:
        raise ValueError(f"No conditions selected from {path}")
    return selected


def normalize_state(stats: dict, key: str, state: np.ndarray) -> np.ndarray:
    item = stats[key]["proprio"]
    low = np.asarray(item["q01"], dtype=np.float32)
    high = np.asarray(item["q99"], dtype=np.float32)
    mask = np.asarray(item["mask"], dtype=bool)
    return np.where(mask, (state - low) * 2 / (high - low + 1e-6) - 1, state)


def unnormalize_action(stats: dict, key: str, action: np.ndarray) -> np.ndarray:
    item = stats[key]["action"]
    low = np.asarray(item["q01"], dtype=np.float32)
    high = np.asarray(item["q99"], dtype=np.float32)
    mask = np.asarray(item["mask"], dtype=bool)
    return np.where(mask, (action + 1) * (high - low + 1e-6) / 2 + low, action)


class OfficialPolicy:
    def __init__(self, checkpoint: str, unnorm_key: str, cfg: float, action_len: int):
        from twinvla.model.twinvla import TwinVLA

        self.model = TwinVLA(pretrained_path=checkpoint, device="cuda:0", dtype=torch.bfloat16)
        self.unnorm_key = unnorm_key
        self.cfg = cfg
        self.action_len = action_len

    def reset(self) -> None:
        return

    def predict(self, observation: dict, instruction: str, _plan_seed: int) -> np.ndarray:
        from TwinVLA.deploy_policy import encode_obs

        obs = encode_obs(observation, instruction)
        return self.model.predict_action(
            self.unnorm_key,
            instruction=obs["instruction"],
            proprio=obs["proprio"],
            image=obs["image"],
            image_wrist_r=obs["image_wrist_r"],
            image_wrist_l=obs["image_wrist_l"],
            action_len=self.action_len,
            cfg=self.cfg,
        )


class NativePolicy:
    def __init__(self, checkpoint: str, unnorm_key: str, cfg: float, action_len: int, denoising_steps: int):
        from commvla.models.dual_arm import CommVLANativeV3Pair

        self.model = CommVLANativeV3Pair.from_pretrained(checkpoint, device="cuda:0", dtype=torch.bfloat16)
        self.model.eval()
        self.stats = json.loads((Path(checkpoint) / "dataset_statistics.json").read_text(encoding="utf-8"))
        self.unnorm_key = unnorm_key
        self.cfg = cfg
        self.action_len = action_len
        self.denoising_steps = denoising_steps

    def reset(self) -> None:
        if hasattr(self.model, "reset_remote_kv_cache"):
            self.model.reset_remote_kv_cache()

    def predict(self, observation: dict, instruction: str, plan_seed: int) -> np.ndarray:
        from TwinVLA.deploy_policy import generate_proprioception

        proprio = generate_proprioception(observation).astype(np.float32)
        normalized = normalize_state(self.stats, self.unnorm_key, proprio)
        visual = observation["observation"]
        inputs = self.model.preprocess_inputs(
            visual["head_camera"]["rgb"][np.newaxis, :].copy(),
            visual["right_camera"]["rgb"][np.newaxis, :].copy(),
            visual["left_camera"]["rgb"][np.newaxis, :].copy(),
            instruction,
            action=None,
        )
        batch = BatchFeature()
        for key, value in inputs.items():
            batch[key] = value.unsqueeze(0) if torch.is_tensor(value) else value
        batch["proprio"] = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        set_seed(plan_seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.model.dtype):
            predicted = self.model.predict_normalized_actions(
                batch,
                cfg=self.cfg,
                num_denoising_steps=self.denoising_steps,
            )
        action = unnormalize_action(self.stats, self.unnorm_key, predicted.detach().cpu().float().numpy())
        return action[0, : self.action_len]


def generate_conditions(args, robotwin_root: Path, output: Path) -> None:
    from commvla.evaluation.robotwin_adapter import find_valid_episode, prepare_robotwin_args, pushd
    from generate_episode_instructions import generate_episode_descriptions

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    now_seed = args.condition_seed_start or 100000 * (1 + args.seed)
    with pushd(robotwin_root):
        base_args, _ = prepare_robotwin_args(
            args.task_name, args.task_config, args.ckpt_setting, "TwinVLA", args.seed
        )
        check_args = dict(base_args)
        check_args["render_freq"] = 0
        for local_id in range(args.generate_conditions):
            rollout_id = args.condition_rollout_offset + local_id
            now_seed, episode_info = find_valid_episode(
                args.task_name, check_args, rollout_id, now_seed
            )
            descriptions = generate_episode_descriptions(args.task_name, [episode_info["info"]], 1)
            candidates = descriptions[0]["unseen"]
            rng = np.random.default_rng(args.seed + rollout_id)
            instruction = str(rng.choice(candidates))
            rows.append({"rollout_id": rollout_id, "env_seed": now_seed, "instruction": instruction})
            write_rows(output, rows)
            print(
                f"[robotwin-conditions] shard_item={local_id + 1}/{args.generate_conditions} "
                f"rollout_id={rollout_id} "
                f"env_seed={now_seed} instruction={instruction!r}",
                flush=True,
            )
            now_seed += 1
    metadata = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "ckpt_setting": args.ckpt_setting,
        "seed": args.seed,
        "condition_seed_start": args.condition_seed_start,
        "condition_rollout_offset": args.condition_rollout_offset,
        "conditions": len(rows),
        "csv": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def evaluate(args, robotwin_root: Path) -> None:
    from commvla.evaluation.robotwin_adapter import (
        class_decorator,
        close_env_safely,
        prepare_robotwin_args,
        pushd,
    )
    from TwinVLA.deploy_policy import convert_to_quat_action

    conditions = read_conditions(Path(args.conditions_csv), args.rollout_offset, args.trials)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_args = None
    with pushd(robotwin_root):
        base_args, _ = prepare_robotwin_args(
            args.task_name, args.task_config, args.ckpt_setting, "TwinVLA", args.seed
        )

    if args.model_kind == "official":
        policy = OfficialPolicy(args.checkpoint, args.unnorm_key, args.cfg, args.action_len)
    else:
        policy = NativePolicy(
            args.checkpoint, args.unnorm_key, args.cfg, args.action_len, args.num_denoising_steps
        )

    rows = []
    plan_seconds_all: list[float] = []
    started = time.time()
    for trial, condition in enumerate(conditions):
        rollout_id = int(condition["rollout_id"])
        env_seed = int(condition["env_seed"])
        instruction = condition["instruction"]
        task_env = None
        success = False
        timeout = False
        error = ""
        local_plan_seconds: list[float] = []
        plan_calls = 0
        ep_started = time.time()
        set_seed(args.seed + rollout_id)
        try:
            with pushd(robotwin_root), rollout_timer(args.max_rollout_seconds):
                task_env = class_decorator(args.task_name)
                task_env.setup_demo(now_ep_num=rollout_id, seed=env_seed, is_test=True, **base_args)
                task_env.set_instruction(instruction=instruction)
                policy.reset()
                while task_env.take_action_cnt < task_env.step_lim:
                    observation = task_env.get_obs()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    plan_started = time.perf_counter()
                    actions = policy.predict(
                        observation,
                        instruction,
                        args.seed + 30000 + rollout_id * 100 + plan_calls,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - plan_started
                    local_plan_seconds.append(elapsed)
                    plan_seconds_all.append(elapsed)
                    plan_calls += 1
                    for action in convert_to_quat_action(actions):
                        task_env.take_action(action, action_type="ee")
                        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                            break
                    if task_env.eval_success:
                        success = True
                        break
        except RolloutTimeout as exc:
            timeout = True
            error = str(exc)
        except Exception:
            error = traceback.format_exc()
            print(f"[robotwin-paired] rollout={rollout_id} failed\n{error}", flush=True)

        steps = int(getattr(task_env, "take_action_cnt", -1)) if task_env is not None else -1
        close_env_safely(
            task_env,
            clear_cache=((trial + 1) % int(base_args.get("clear_cache_freq", 5)) == 0),
        )
        row = {
            "rollout_id": rollout_id,
            "env_seed": env_seed,
            "instruction": instruction,
            "success": int(success),
            "steps": steps,
            "plan_calls": plan_calls,
            "episode_seconds": time.time() - ep_started,
            "plan_seconds_mean": float(np.mean(local_plan_seconds)) if local_plan_seconds else 0.0,
            "plan_seconds_p95": percentile(local_plan_seconds, 95),
            "timeout": int(timeout),
            "error": error,
        }
        rows.append(row)
        write_rows(out_dir / "rollouts.csv", rows)
        successes = sum(item["success"] for item in rows)
        print(
            f"[robotwin-paired][{args.model_kind}] trial={trial + 1}/{len(conditions)} "
            f"rollout={rollout_id} success={int(success)} rate={successes / len(rows):.3f} "
            f"timeout={int(timeout)}",
            flush=True,
        )

    success_count = sum(row["success"] for row in rows)
    summary = {
        "model_kind": args.model_kind,
        "checkpoint": args.checkpoint,
        "task_name": args.task_name,
        "conditions_csv": args.conditions_csv,
        "trials": len(rows),
        "success_count": success_count,
        "success_rate": success_count / len(rows),
        "timeout_count": sum(row["timeout"] for row in rows),
        "error_count": sum(bool(row["error"]) for row in rows),
        "cfg": args.cfg,
        "num_denoising_steps": args.num_denoising_steps,
        "action_len": args.action_len,
        "plan_latency": {
            "count": len(plan_seconds_all),
            "mean": float(np.mean(plan_seconds_all)) if plan_seconds_all else 0.0,
            "p50": percentile(plan_seconds_all, 50),
            "p95": percentile(plan_seconds_all, 95),
            "max": float(np.max(plan_seconds_all)) if plan_seconds_all else 0.0,
        },
        "episode_seconds_mean": float(np.mean([row["episode_seconds"] for row in rows])),
        "seconds": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--model-kind", choices=["official", "native"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--task-name", default="handover_block")
    parser.add_argument("--unnorm-key", default="robotwin_handover_block")
    parser.add_argument("--task-config", default="demo_clean_novideo")
    parser.add_argument("--ckpt-setting", default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=1.1)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--action-len", type=int, default=20)
    parser.add_argument("--conditions-csv")
    parser.add_argument("--generate-conditions", type=int, default=0)
    parser.add_argument("--condition-rollout-offset", type=int, default=0)
    parser.add_argument("--condition-seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--rollout-offset", type=int, default=0)
    parser.add_argument("--max-rollout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    robotwin_root = Path(args.robotwin_root).resolve()
    add_robotwin_paths(robotwin_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    if args.generate_conditions:
        if not args.conditions_csv:
            parser.error("--conditions-csv is required with --generate-conditions")
        generate_conditions(args, robotwin_root, Path(args.conditions_csv))
        return
    for required in ("model_kind", "checkpoint", "conditions_csv", "output_dir"):
        if not getattr(args, required):
            parser.error(f"--{required.replace('_', '-')} is required for evaluation")
    evaluate(args, robotwin_root)


if __name__ == "__main__":
    main()
