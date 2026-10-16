"""Two-process Tabletop rollout smoke for CommVLA-native v3.5.

Rank 0 owns the Tabletop environment. At every planning step it broadcasts the
current observation payload to rank 1. Each rank runs one SingleVLA-side agent,
exchanges remote K/V through the distributed v3.5 forward path, decodes its own
10D action chunk, and rank 0 concatenates both chunks before stepping the env.
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
import torch.distributed as dist
from dm_env import StepType as st
from transformers.feature_extraction_utils import BatchFeature

from twinvla.model.singlevla import SingleVLA
from commvla.communication import ChannelConfig, PeerCommunicationChannel, get_peer_channel, set_peer_channel

from .tabletop_two_process_core import (
    action_token_from_private,
    distributed_v35_forward,
    fuse_distributed_action_token,
    inference_autocast,
    load_distributed_extra_modules,
    log,
    make_side_batch,
    module_dtype,
    parse_dtype,
    set_seed,
    unnormalize_action,
)


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


def _obs_payload(
    obs: dict,
    plan_seed: int,
    *,
    episode_id: int,
    planning_round: int,
) -> dict:
    return {
        "done": False,
        "ee_6d_pos": np.asarray(obs["ee_6d_pos"], dtype=np.float32),
        "language_instruction": obs["language_instruction"],
        "images": {
            "back": np.asarray(obs["images"]["back"]),
            "wrist_left": np.asarray(obs["images"]["wrist_left"]),
            "wrist_right": np.asarray(obs["images"]["wrist_right"]),
        },
        "plan_seed": int(plan_seed),
        "episode_id": int(episode_id),
        "planning_round": int(planning_round),
    }


def _broadcast_payload(payload: dict | None) -> dict:
    holder = [payload]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def _save_plan_payload(payload: dict, output_dir: str | None) -> None:
    if not output_dir:
        return
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"episode_{int(payload['episode_id']):04d}_round_{int(payload['planning_round']):03d}.npz"
    np.savez_compressed(
        target,
        ee_6d_pos=np.asarray(payload["ee_6d_pos"], dtype=np.float32),
        language_instruction=np.asarray(str(payload["language_instruction"])),
        back=np.asarray(payload["images"]["back"]),
        wrist_left=np.asarray(payload["images"]["wrist_left"]),
        wrist_right=np.asarray(payload["images"]["wrist_right"]),
        plan_seed=np.asarray(int(payload["plan_seed"]), dtype=np.int64),
        episode_id=np.asarray(int(payload["episode_id"]), dtype=np.int64),
        planning_round=np.asarray(int(payload["planning_round"]), dtype=np.int64),
    )


def _parse_rollout_ids(args) -> list[int]:
    if args.rollout_ids and args.rollout_ids_file:
        raise ValueError("Use only one of --rollout-ids and --rollout-ids-file")
    if args.rollout_ids:
        return [int(item.strip()) for item in args.rollout_ids.split(",") if item.strip()]
    if args.rollout_ids_file:
        text = Path(args.rollout_ids_file).read_text(encoding="utf-8")
        ids: list[int] = []
        for raw in text.replace(",", "\n").splitlines():
            item = raw.strip()
            if not item or item.startswith("#"):
                continue
            ids.append(int(item))
        return ids
    return [args.rollout_offset + trial for trial in range(args.trials)]


def _to_side_batch(agent, stats: dict, unnorm_key: str, payload: dict, side: str, device, dtype: torch.dtype) -> BatchFeature:
    obs = {
        "ee_6d_pos": payload["ee_6d_pos"],
        "language_instruction": payload["language_instruction"],
        "images": payload["images"],
    }
    return make_side_batch(agent, stats, unnorm_key, obs, side, device, dtype)


def _predict_plan(
    agent,
    stats: dict,
    args,
    payload: dict,
    side: str,
    device,
    dtype: torch.dtype,
    common_kv_mode: str,
    extra_modules: dict[str, object],
    common_sync_strategy: str = "full",
    common_sync_layers: str = "all",
):
    channel = get_peer_channel()
    if channel is not None:
        channel.begin_round(payload["episode_id"], payload["planning_round"])
    batch = _to_side_batch(agent, stats, args.unnorm_key, payload, side, device, dtype)
    started = time.perf_counter()
    with torch.no_grad(), inference_autocast(dtype):
        set_seed(int(payload["plan_seed"]) + dist.get_rank())
        hidden_private, private_modal = distributed_v35_forward(
            agent,
            batch,
            side,
            common_kv_mode=common_kv_mode,
            common_layer_gate=extra_modules.get("common_layer_gate"),
            private_remote_kv_mode=args.private_remote_kv_mode,
            common_sync_strategy=common_sync_strategy,
            common_sync_layers=common_sync_layers,
        )
        token = action_token_from_private(agent, hidden_private, private_modal)
        extra_modules["action_token_ablation_mode"] = args.action_token_ablation_mode
        token = fuse_distributed_action_token(token, extra_modules)
        head = agent.action_head
        head_dtype = module_dtype(head)
        normalized_side = head.denoise(
            token.to(dtype=head_dtype),
            batch["proprio"][:, 0, :].to(dtype=head_dtype),
            denoising_steps=args.num_denoising_steps,
            cfg=args.cfg,
        ).reshape(-1, int(agent.config.action_len), int(agent.config.action_dim))
    torch.cuda.synchronize(device)
    plan_seconds = time.perf_counter() - started

    local_cpu = normalized_side.detach().contiguous().cpu()
    gathered = [torch.empty_like(local_cpu) for _ in range(2)]
    dist.all_gather(gathered, local_cpu)
    local_finite = bool(torch.isfinite(normalized_side).all().item())
    finite_flags = [None, None]
    dist.all_gather_object(finite_flags, local_finite)
    if dist.get_rank() != 0:
        return None, plan_seconds, finite_flags
    normalized = torch.cat([gathered[0], gathered[1]], dim=-1).float().numpy()
    actions = unnormalize_action(stats, args.unnorm_key, normalized)[0]
    return actions, plan_seconds, finite_flags


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
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--rollout-offset", type=int, default=0)
    parser.add_argument("--rollout-ids", default=None, help="Comma-separated benchmark rollout ids to run.")
    parser.add_argument("--rollout-ids-file", default=None, help="Text file with one benchmark rollout id per line.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-seed-offset", type=int, default=0, help="Offset applied only to action denoising plan seeds.")
    parser.add_argument("--cfg", type=float, default=1.1)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--backend", choices=["gloo"], default="gloo")
    parser.add_argument("--max-rollout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--save-plan-payload-dir",
        default=None,
        help="Optional directory for compressed planning-boundary observations.",
    )
    parser.add_argument(
        "--action-token-ablation-mode",
        choices=["none", "zero_remote", "random_remote", "stale_remote", "disable_fusion"],
        default="none",
    )
    parser.add_argument(
        "--private-remote-kv-mode",
        choices=["checkpoint", "full", "no_remote", "random_remote"],
        default="checkpoint",
    )
    parser.add_argument(
        "--common-kv-mode",
        choices=["checkpoint", "local", "left_shared", "right_shared", "avg_shared", "sync_qkv_avg_mlp_avg"],
        default="checkpoint",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comm-profile-dir", default=None)
    parser.add_argument("--comm-profile-mode", choices=["none", "bytes", "timing"], default="none")
    parser.add_argument("--comm-action-refresh", choices=["1", "2", "4", "8", "initial"], default="1")
    parser.add_argument("--comm-common-refresh", choices=["1", "2", "4", "8", "initial"], default="1")
    parser.add_argument("--comm-remote-kv-refresh", choices=["1", "2", "4", "8", "initial"], default="1")
    codec_choices = [
        "raw", "int8", "int8_packed", "fp8_e4m3", "fp8_e5m2", "int4",
        "mixed_i4_a", "mixed_i4_b", "delta_int8", "delta_int4",
    ]
    parser.add_argument("--comm-action-codec", choices=codec_choices, default="raw")
    parser.add_argument("--comm-common-codec", choices=codec_choices, default="raw")
    parser.add_argument("--comm-remote-kv-codec", choices=codec_choices, default="raw")
    parser.add_argument("--comm-delta-keyframe", type=int, default=4)
    parser.add_argument("--comm-common-sync-strategy", choices=["full", "hidden_only"], default="full")
    parser.add_argument(
        "--comm-common-sync-layers",
        default="all",
        help="Named mask (all/late/even/late_even/last6), ids:0,2,..., or bits:<24 bits>.",
    )
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = parse_dtype(args.dtype)
    set_seed(args.seed)

    communication_enabled = bool(
        args.comm_profile_dir
        or args.comm_profile_mode != "none"
        or args.comm_action_refresh != "1"
        or args.comm_common_refresh != "1"
        or args.comm_remote_kv_refresh != "1"
        or args.comm_action_codec != "raw"
        or args.comm_common_codec != "raw"
        or args.comm_remote_kv_codec != "raw"
        or args.comm_common_sync_strategy != "full"
        or args.comm_common_sync_layers != "all"
    )
    channel = None
    if communication_enabled:
        channel = PeerCommunicationChannel(
            ChannelConfig.from_values(
                action_refresh=args.comm_action_refresh,
                common_refresh=args.comm_common_refresh,
                remote_kv_refresh=args.comm_remote_kv_refresh,
                action_codec=args.comm_action_codec,
                common_codec=args.comm_common_codec,
                remote_kv_codec=args.comm_remote_kv_codec,
                delta_keyframe=args.comm_delta_keyframe,
                profile_mode=args.comm_profile_mode,
            ),
            Path(args.comm_profile_dir) if args.comm_profile_dir else None,
        )
        set_peer_channel(channel)

    checkpoint = Path(args.checkpoint)
    native_config = json.loads((checkpoint / "commvla_native_v3_config.json").read_text(encoding="utf-8"))
    if args.private_remote_kv_mode == "checkpoint":
        args.private_remote_kv_mode = str(native_config.get("remote_kv_mode", "full"))
    common_kv_mode = str(native_config.get("common_kv_mode", "right_shared"))
    if args.common_kv_mode != "checkpoint":
        common_kv_mode = args.common_kv_mode
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text(encoding="utf-8"))
    side = "left" if rank == 0 else "right"
    agent_path = checkpoint / ("left_private_agent" if rank == 0 else "right_private_agent")
    log(f"loading {side} agent from {agent_path}")
    agent = SingleVLA(pretrained_path=str(agent_path), device=device, dtype=dtype).model
    agent.config.use_cache = False
    agent.eval()
    extra_modules = load_distributed_extra_modules(native_config, checkpoint, agent, device, dtype, side)
    log(f"loaded {side} agent")

    rows = []
    if rank == 0:
        import tabletop

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        env = tabletop.env(args.task_name, args.action_space)
    else:
        out_dir = None
        env = None

    rollout_ids = _parse_rollout_ids(args)
    args.trials = len(rollout_ids)

    for trial, rollout_id in enumerate(rollout_ids):
        if rank == 0:
            set_seed(args.seed + rollout_id)
            ts = env.reset()
            if args.benchmark:
                ts = env.task.benchmark_init(env.physics, rollout_id)
            rewards = []
            actions = None
            action_counter = 0
            plan_history = []
            plan_calls = 0
            plan_seconds = []
            finite_history = []
            timeout = False
            timeout_message = ""
            ep_started = time.time()
            try:
                with rollout_timer(args.max_rollout_seconds):
                    while True:
                        obs = ts.observation
                        step_idx = len(rewards)
                        if args.execution_mode == "temporal_agg":
                            payload = _obs_payload(
                                obs,
                                args.seed + args.plan_seed_offset + 30000 + rollout_id * 100 + plan_calls,
                                episode_id=rollout_id,
                                planning_round=plan_calls,
                            )
                            _save_plan_payload(payload, args.save_plan_payload_dir)
                            _broadcast_payload(payload)
                            current_actions, elapsed, finite_flags = _predict_plan(
                                agent,
                                stats,
                                args,
                                payload,
                                side,
                                device,
                                dtype,
                                common_kv_mode,
                                extra_modules,
                                args.comm_common_sync_strategy,
                                args.comm_common_sync_layers,
                            )
                            plan_seconds.append(elapsed)
                            finite_history.append(finite_flags)
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
                                payload = _obs_payload(
                                    obs,
                                    args.seed + args.plan_seed_offset + 30000 + rollout_id * 100 + plan_calls,
                                    episode_id=rollout_id,
                                    planning_round=plan_calls,
                                )
                                _save_plan_payload(payload, args.save_plan_payload_dir)
                                _broadcast_payload(payload)
                                actions, elapsed, finite_flags = _predict_plan(
                                    agent,
                                    stats,
                                    args,
                                    payload,
                                    side,
                                    device,
                                    dtype,
                                    common_kv_mode,
                                    extra_modules,
                                    args.comm_common_sync_strategy,
                                    args.comm_common_sync_layers,
                                )
                                plan_seconds.append(elapsed)
                                finite_history.append(finite_flags)
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
            finally:
                _broadcast_payload({"done": True})
            numeric_rewards = np.array([r for r in rewards if r is not None], dtype=np.float32)
            highest = float(numeric_rewards.max()) if numeric_rewards.size else 0.0
            row = {
                "rollout_id": rollout_id,
                "highest_reward": highest,
                "success": int(highest == env.task.max_reward),
                "steps": len(rewards),
                "plan_calls": plan_calls,
                "episode_seconds": time.time() - ep_started,
                "plan_seconds_mean": float(np.mean(plan_seconds)) if plan_seconds else 0.0,
                "plan_seconds_p95": float(np.percentile(np.asarray(plan_seconds), 95)) if plan_seconds else 0.0,
                "finite_all": int(all(all(flags) for flags in finite_history)),
                "timeout": int(timeout),
                "timeout_message": timeout_message,
            }
            rows.append(row)
            if channel is not None:
                channel.flush()
            print(f"[distributed-v35-rollout] trial={trial + 1}/{args.trials} row={row}", flush=True)
        else:
            while True:
                payload = _broadcast_payload(None)
                if payload.get("done"):
                    break
                _predict_plan(
                    agent, stats, args, payload, side, device, dtype, common_kv_mode, extra_modules,
                    args.comm_common_sync_strategy, args.comm_common_sync_layers,
                )

    communication_stats_by_rank = [None, None]
    if channel is not None:
        dist.all_gather_object(communication_stats_by_rank, channel.summary())

    if rank == 0:
        csv_path = out_dir / "rollouts.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "checkpoint": str(checkpoint),
            "dtype": args.dtype,
            "backend": args.backend,
            "common_kv_mode": common_kv_mode,
            "common_fusion_mode": extra_modules.get("common_fusion_mode"),
            "action_token_fusion_mode": extra_modules.get("action_token_fusion_mode"),
            "action_token_ablation_mode": args.action_token_ablation_mode,
            "private_remote_kv_mode": args.private_remote_kv_mode,
            "communication": {
                "enabled": communication_enabled,
                "profile_mode": args.comm_profile_mode,
                "action_refresh": args.comm_action_refresh,
                "common_refresh": args.comm_common_refresh,
                "remote_kv_refresh": args.comm_remote_kv_refresh,
                "action_codec": args.comm_action_codec,
                "common_codec": args.comm_common_codec,
                "remote_kv_codec": args.comm_remote_kv_codec,
                "delta_keyframe": args.comm_delta_keyframe,
                "common_sync_strategy": args.comm_common_sync_strategy,
                "common_sync_layers": args.comm_common_sync_layers,
            },
            "communication_stats_by_rank": communication_stats_by_rank if channel is not None else None,
            "task_name": args.task_name,
            "trials": args.trials,
            "rollout_ids": rollout_ids,
            "plan_seed_offset": args.plan_seed_offset,
            "seed": args.seed,
            "cfg": args.cfg,
            "num_denoising_steps": args.num_denoising_steps,
            "successes": int(sum(row["success"] for row in rows)),
            "timeouts": int(sum(row["timeout"] for row in rows)),
            "mean_plan_seconds": float(np.mean([row["plan_seconds_mean"] for row in rows])) if rows else 0.0,
            "execution_mode": args.execution_mode,
            "action_len": args.action_len,
            "execute_chunk_size": args.execute_chunk_size,
            "temporal_agg_decay": args.temporal_agg_decay,
            "temporal_agg_max_plans": args.temporal_agg_max_plans,
            "rows": rows,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    if channel is not None:
        channel.close()
        set_peer_channel(None)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
