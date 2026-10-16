"""Single-process Tabletop-Sim rollout for CommVLA-native v3 checkpoints."""

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
from transformers.feature_extraction_utils import BatchFeature

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


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def normalize_state(stats: dict, unnorm_key: str, state: np.ndarray) -> np.ndarray:
    item = stats[unnorm_key]["proprio"]
    low = _array(item["q01"])
    high = _array(item["q99"])
    mask = np.asarray(item["mask"], dtype=bool)
    return np.where(mask, (state - low) * 2 / (high - low + 1e-6) - 1, state)


def unnormalize_action(stats: dict, unnorm_key: str, action: np.ndarray) -> np.ndarray:
    item = stats[unnorm_key]["action"]
    low = _array(item["q01"])
    high = _array(item["q99"])
    mask = np.asarray(item["mask"], dtype=bool)
    return np.where(mask, (action + 1) * (high - low + 1e-6) / 2 + low, action)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def make_native_batch(model: CommVLANativeV3Pair, stats: dict, *, unnorm_key: str, obs: dict) -> BatchFeature:
    normalized_proprio = normalize_state(stats, unnorm_key, obs["ee_6d_pos"])
    inputs = model.preprocess_inputs(
        obs["images"]["back"][np.newaxis, :].copy(),
        obs["images"]["wrist_right"][np.newaxis, :].copy(),
        obs["images"]["wrist_left"][np.newaxis, :].copy(),
        obs["language_instruction"],
        action=None,
    )
    batch = BatchFeature()
    for key, value in inputs.items():
        batch[key] = value.unsqueeze(0) if torch.is_tensor(value) else value
    batch["proprio"] = torch.tensor(normalized_proprio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return batch


def predict_action(model: CommVLANativeV3Pair, stats: dict, args, obs: dict, plan_seed: int):
    batch = make_native_batch(model, stats, unnorm_key=args.unnorm_key, obs=obs)
    with torch.no_grad(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=model.dtype):
        set_seed(plan_seed)
        normalized_action = model.predict_normalized_actions(batch, cfg=args.cfg, num_denoising_steps=args.num_denoising_steps)
    normalized = normalized_action.detach().cpu().float().numpy()
    action = unnormalize_action(stats, args.unnorm_key, normalized)
    return action[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-name", default="aloha_handover_box")
    parser.add_argument("--unnorm-key", default="aloha_handover_box")
    parser.add_argument("--action-space", default="ee_6d_pos")
    parser.add_argument("--action-len", type=int, default=20)
    parser.add_argument("--execution-mode", choices=["chunk", "partial", "receding", "temporal_agg"], default="chunk")
    parser.add_argument("--execute-chunk-size", type=int, default=None)
    parser.add_argument("--temporal-agg-decay", type=float, default=0.5)
    parser.add_argument("--temporal-agg-max-plans", type=int, default=20)
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
    parser.add_argument("--max-rollout-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import tabletop

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((Path(args.checkpoint) / "dataset_statistics.json").read_text(encoding="utf-8"))
    model = CommVLANativeV3Pair.from_pretrained(args.checkpoint, device="cuda:0" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
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
        plan_history = []
        plan_calls = 0
        local_plan_seconds: list[float] = []
        instruction = None
        timeout = False
        timeout_message = ""
        ep_started = time.time()
        try:
            with rollout_timer(args.max_rollout_seconds):
                while True:
                    obs = ts.observation
                    instruction = obs["language_instruction"]
                    step_idx = len(rewards)
                    if args.execution_mode == "temporal_agg":
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        plan_started = time.perf_counter()
                        current_actions = predict_action(
                            model,
                            stats,
                            args,
                            obs,
                            args.seed + 30000 + global_rollout_id * 10000 + plan_calls,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        plan_elapsed = time.perf_counter() - plan_started
                        local_plan_seconds.append(plan_elapsed)
                        plan_seconds_all.append(plan_elapsed)
                        plan_history.append((step_idx, current_actions))
                        if args.temporal_agg_max_plans > 0:
                            plan_history = plan_history[-args.temporal_agg_max_plans :]
                        plan_calls += 1
                        candidates = []
                        weights = []
                        for plan_start, plan_actions in plan_history:
                            action_idx = step_idx - plan_start
                            if 0 <= action_idx < args.action_len:
                                candidates.append(plan_actions[action_idx])
                                weights.append(np.exp(-args.temporal_agg_decay * action_idx))
                        if not candidates:
                            raise RuntimeError("temporal_agg has no valid action candidates")
                        weights_np = np.asarray(weights, dtype=np.float32)
                        weights_np = weights_np / (weights_np.sum() + 1e-8)
                        action = np.sum(np.stack(candidates, axis=0) * weights_np[:, None], axis=0)
                    else:
                        if action_counter == 0:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            plan_started = time.perf_counter()
                            actions = predict_action(model, stats, args, obs, args.seed + 30000 + global_rollout_id * 100 + plan_calls)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            plan_elapsed = time.perf_counter() - plan_started
                            local_plan_seconds.append(plan_elapsed)
                            plan_seconds_all.append(plan_elapsed)
                            plan_calls += 1
                        action = actions[action_counter]
                    ts = env.step(action)
                    rewards.append(ts.reward)
                    if args.execution_mode == "temporal_agg":
                        action_counter = 0
                    else:
                        action_counter += 1
                        if args.execution_mode == "receding":
                            replan_interval = 1
                        elif args.execution_mode == "partial":
                            replan_interval = args.execute_chunk_size or args.action_len
                            replan_interval = max(1, min(replan_interval, args.action_len))
                        else:
                            replan_interval = args.action_len
                        if action_counter == replan_interval:
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
            f"[commvla-native-v3] rollout={rollout_id + 1}/{args.trials} highest={highest} "
            f"success_rate={np.mean(np.array(highest_rewards) == env.task.max_reward):.3f} timeout={int(timeout)}",
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
        "execution_mode": args.execution_mode,
        "execute_chunk_size": args.execute_chunk_size,
        "temporal_agg_decay": args.temporal_agg_decay,
        "temporal_agg_max_plans": args.temporal_agg_max_plans,
        "max_rollout_seconds": args.max_rollout_seconds,
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
