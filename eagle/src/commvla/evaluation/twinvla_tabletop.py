"""Deterministic Tabletop-Sim evaluation for official TwinVLA checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path


def parse_rollout_ids(value: str | None, ids_file: str | None, trials: int, offset: int) -> list[int]:
    if value and ids_file:
        raise ValueError("Use either --rollout-ids or --rollout-ids-file")
    if value:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif ids_file:
        ids = [int(line.strip()) for line in Path(ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        ids = list(range(offset, offset + trials))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Rollout IDs must be non-empty and unique")
    return ids


def planning_seed(base_seed: int, plan_seed_offset: int, rollout_id: int, plan_call: int) -> int:
    return base_seed + plan_seed_offset + 30000 + rollout_id * 100 + plan_call


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class RolloutTimeout(RuntimeError):
    pass


@contextmanager
def _rollout_timer(seconds: float):
    if seconds <= 0:
        yield
        return

    def handler(_signum, _frame):
        raise RolloutTimeout(f"rollout exceeded {seconds:.1f}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--action-space", default="ee_6d_pos")
    parser.add_argument("--action-len", type=int, default=20)
    parser.add_argument("--execution-mode", choices=["chunk"], default="chunk")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--rollout-offset", type=int, default=0)
    parser.add_argument("--rollout-ids", default=None)
    parser.add_argument("--rollout-ids-file", default=None)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--plan-seed-offset", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=1.1)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-rollout-seconds", type=float, default=180.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from dm_env import StepType as st
    from twinvla.model.twinvla import TwinVLA
    import tabletop

    if args.num_denoising_steps != 10:
        raise ValueError("Official TwinVLA wrapper uses the frozen 10-step denoising protocol")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _set_seed(args.seed)
    model = TwinVLA(pretrained_path=args.checkpoint, device=device, dtype=dtype)
    env = tabletop.env(args.task_name, args.action_space)
    rollout_ids = parse_rollout_ids(args.rollout_ids, args.rollout_ids_file, args.trials, args.rollout_offset)
    rows: list[dict[str, object]] = []

    for rollout_id in rollout_ids:
        _set_seed(args.seed + rollout_id)
        ts = env.reset()
        if args.benchmark:
            ts = env.task.benchmark_init(env.physics, rollout_id)
        rewards = []
        actions = None
        action_counter = 0
        plan_calls = 0
        plan_seconds = []
        finite_flags = []
        timeout = False
        timeout_message = ""
        episode_started = time.time()
        try:
            with _rollout_timer(args.max_rollout_seconds), torch.inference_mode():
                while True:
                    obs = ts.observation
                    if action_counter == 0:
                        _set_seed(planning_seed(args.seed, args.plan_seed_offset, rollout_id, plan_calls))
                        started = time.time()
                        actions = model.predict_action(
                            unnorm_key=args.unnorm_key,
                            instruction=obs["language_instruction"],
                            image=obs["images"]["back"],
                            image_wrist_r=obs["images"]["wrist_right"],
                            image_wrist_l=obs["images"]["wrist_left"],
                            proprio=obs["ee_6d_pos"],
                            action_len=args.action_len,
                            cfg=args.cfg,
                        )
                        torch.cuda.synchronize()
                        plan_seconds.append(time.time() - started)
                        finite_flags.append(bool(np.isfinite(actions).all()))
                        plan_calls += 1
                    ts = env.step(actions[action_counter])
                    rewards.append(ts.reward)
                    action_counter = (action_counter + 1) % args.action_len
                    if ts.reward == env.task.max_reward or ts.step_type == st.LAST:
                        break
        except RolloutTimeout as exc:
            timeout = True
            timeout_message = str(exc)
        numeric_rewards = np.asarray([value for value in rewards if value is not None], dtype=np.float32)
        highest = float(numeric_rewards.max()) if numeric_rewards.size else 0.0
        rows.append(
            {
                "rollout_id": rollout_id,
                "highest_reward": highest,
                "success": int(highest == env.task.max_reward),
                "steps": len(rewards),
                "plan_calls": plan_calls,
                "episode_seconds": time.time() - episode_started,
                "plan_seconds_mean": float(np.mean(plan_seconds)) if plan_seconds else 0.0,
                "plan_seconds_p95": float(np.percentile(plan_seconds, 95)) if plan_seconds else 0.0,
                "finite_all": int(all(finite_flags)),
                "timeout": int(timeout),
                "timeout_message": timeout_message,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "rollouts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "kind": "twinvla",
        "checkpoint": str(Path(args.checkpoint)),
        "task_name": args.task_name,
        "trials": len(rows),
        "rollout_ids": rollout_ids,
        "seed": args.seed,
        "plan_seed_offset": args.plan_seed_offset,
        "cfg": args.cfg,
        "num_denoising_steps": args.num_denoising_steps,
        "dtype": args.dtype,
        "execution_mode": args.execution_mode,
        "action_len": args.action_len,
        "successes": sum(int(row["success"]) for row in rows),
        "timeouts": sum(int(row["timeout"]) for row in rows),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

