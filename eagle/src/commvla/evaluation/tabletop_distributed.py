"""Dual-device Tabletop-Sim rollout for CommVLA-native v3 checkpoints.

This runner keeps the v3 model semantics unchanged but places the left and
right SingleVLA agents on separate CUDA devices. It records action-plan latency
so we can compare split-device inference against the single-device rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from dm_env import StepType as st

from commvla.evaluation.tabletop import make_native_batch, unnormalize_action
from commvla.models.dual_arm import CommVLANativeV3Pair


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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_devices(model: CommVLANativeV3Pair) -> None:
    if not torch.cuda.is_available():
        return
    devices = set()
    for value in (model.left_device, model.right_device):
        if isinstance(value, str) and value.startswith("cuda"):
            devices.add(torch.device(value).index or 0)
        elif isinstance(value, int):
            devices.add(value)
    for idx in sorted(devices):
        torch.cuda.synchronize(idx)


def parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@contextmanager
def inference_autocast(dtype: torch.dtype):
    if not torch.cuda.is_available() or dtype == torch.float32:
        with torch.amp.autocast("cuda", enabled=False):
            yield
    else:
        with torch.autocast("cuda", dtype=dtype):
            yield


def predict_action_timed(model: CommVLANativeV3Pair, stats: dict, args, obs: dict, plan_seed: int):
    batch = make_native_batch(model, stats, unnorm_key=args.unnorm_key, obs=obs)
    sync_devices(model)
    started = time.perf_counter()
    with torch.no_grad(), inference_autocast(model.dtype):
        set_seed(plan_seed)
        normalized_action = model.predict_normalized_actions(batch, cfg=args.cfg, num_denoising_steps=args.num_denoising_steps)
    sync_devices(model)
    plan_seconds = time.perf_counter() - started
    normalized = normalized_action.detach().cpu().float().numpy()
    action = unnormalize_action(stats, args.unnorm_key, normalized)
    return action[0], plan_seconds


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-name", default="aloha_handover_box")
    parser.add_argument("--unnorm-key", default="aloha_handover_box")
    parser.add_argument("--action-space", default="ee_6d_pos")
    parser.add_argument("--action-len", type=int, default=20)
    parser.add_argument("--execution-mode", choices=["chunk"], default="chunk")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rollout-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=1.1)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--remote-kv-mode", choices=["checkpoint", "full", "no_remote", "random_remote", "stale_remote"], default="checkpoint")
    parser.add_argument(
        "--common-kv-mode",
        choices=["checkpoint", "local", "left_shared", "right_shared", "avg_shared", "sync_qkv_avg_mlp_avg"],
        default="checkpoint",
    )
    parser.add_argument("--left-device", default="cuda:0")
    parser.add_argument("--right-device", default="cuda:1")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-rollout-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import tabletop

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((Path(args.checkpoint) / "dataset_statistics.json").read_text(encoding="utf-8"))
    model = CommVLANativeV3Pair.from_pretrained(
        args.checkpoint,
        device=args.left_device if torch.cuda.is_available() else "cpu",
        right_device=args.right_device if torch.cuda.is_available() else None,
        dtype=parse_dtype(args.dtype),
    )
    if args.remote_kv_mode != "checkpoint":
        model.native_config.remote_kv_mode = args.remote_kv_mode
    if args.common_kv_mode != "checkpoint":
        model.native_config.common_kv_mode = args.common_kv_mode
    model.eval()
    env = tabletop.env(args.task_name, args.action_space)

    rows = []
    highest_rewards = []
    plan_seconds_all: list[float] = []
    started = time.time()
    for rollout_id in range(args.trials):
        global_rollout_id = args.rollout_offset + rollout_id
        set_seed(args.seed + global_rollout_id)
        if hasattr(model, "reset_remote_kv_cache"):
            model.reset_remote_kv_cache()
        ts = env.reset()
        if args.benchmark:
            ts = env.task.benchmark_init(env.physics, global_rollout_id)
        rewards = []
        actions = None
        action_counter = 0
        plan_calls = 0
        instruction = None
        timeout = False
        timeout_message = ""
        local_plan_seconds: list[float] = []
        ep_started = time.time()
        try:
            with rollout_timer(args.max_rollout_seconds):
                while True:
                    obs = ts.observation
                    instruction = obs["language_instruction"]
                    if action_counter == 0:
                        actions, plan_seconds = predict_action_timed(
                            model,
                            stats,
                            args,
                            obs,
                            args.seed + 30000 + global_rollout_id * 100 + plan_calls,
                        )
                        local_plan_seconds.append(plan_seconds)
                        plan_seconds_all.append(plan_seconds)
                        plan_calls += 1
                    ts = env.step(actions[action_counter])
                    rewards.append(ts.reward)
                    action_counter += 1
                    if action_counter == args.action_len:
                        action_counter = 0
                    if ts.reward == env.task.max_reward or ts.step_type == st.LAST:
                        break
        except RolloutTimeout as exc:
            timeout = True
            timeout_message = str(exc)
        numeric_rewards = np.array([r for r in rewards if r is not None], dtype=np.float32)
        highest = float(numeric_rewards.max()) if numeric_rewards.size else 0.0
        highest_rewards.append(highest)
        row = {
            "rollout_id": global_rollout_id,
            "instruction": instruction,
            "highest_reward": highest,
            "return": float(numeric_rewards.sum()) if numeric_rewards.size else 0.0,
            "success": int(highest == env.task.max_reward),
            "steps": len(rewards),
            "plan_calls": plan_calls,
            "episode_seconds": time.time() - ep_started,
            "plan_seconds_mean": float(np.mean(local_plan_seconds)) if local_plan_seconds else 0.0,
            "plan_seconds_p95": percentile(local_plan_seconds, 95),
            "timeout": int(timeout),
            "timeout_message": timeout_message,
        }
        rows.append(row)
        print(
            f"[commvla-native-v3-dual] rollout={rollout_id + 1}/{args.trials} highest={highest} "
            f"success_rate={np.mean(np.array(highest_rewards) == env.task.max_reward):.3f} "
            f"plan_mean={row['plan_seconds_mean']:.4f}s timeout={int(timeout)}",
            flush=True,
        )

    with (out_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    highest_arr = np.array(highest_rewards)
    summary = {
        "checkpoint": args.checkpoint,
        "task_name": args.task_name,
        "trials": args.trials,
        "seed": args.seed,
        "cfg": args.cfg,
        "benchmark": args.benchmark,
        "remote_kv_mode": model.native_config.remote_kv_mode,
        "common_kv_mode": model.native_config.common_kv_mode,
        "left_device": args.left_device,
        "right_device": args.right_device,
        "dtype": args.dtype,
        "success_count": int((highest_arr == env.task.max_reward).sum()),
        "success_rate": float(np.mean(highest_arr == env.task.max_reward)),
        "timeout_count": int(sum(row["timeout"] for row in rows)),
        "env_max_reward": int(env.task.max_reward),
        "reward_ge": {str(r): int((highest_arr >= r).sum()) for r in range(env.task.max_reward + 1)},
        "plan_latency": {
            "count": len(plan_seconds_all),
            "mean": float(np.mean(plan_seconds_all)) if plan_seconds_all else 0.0,
            "p50": percentile(plan_seconds_all, 50),
            "p95": percentile(plan_seconds_all, 95),
            "max": float(np.max(plan_seconds_all)) if plan_seconds_all else 0.0,
        },
        "episode_seconds_mean": float(np.mean([row["episode_seconds"] for row in rows])) if rows else 0.0,
        "seconds": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
