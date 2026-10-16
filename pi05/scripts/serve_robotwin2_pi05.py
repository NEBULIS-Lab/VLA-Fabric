#!/usr/bin/env python3
"""Serve one restored PI0.5 V2 checkpoint to a RoboTwin process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
import traceback

import numpy as np

from pi05_fabric.evaluation.gpu_claim import claim_jax_device
from pi05_fabric.evaluation.robotwin2_diagnostic import action_support_summary
from pi05_fabric.evaluation.robotwin2_diagnostic import effective_interaction
from pi05_fabric.evaluation.robotwin2_ipc import receive_message
from pi05_fabric.evaluation.robotwin2_ipc import send_message
from pi05_fabric.evaluation.robotwin2_policy import load_robotwin_pi05_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-stage", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=20260824)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument(
        "--inference-profile",
        choices=("full", "no_common", "no_private", "core", "all_off", "common_only"),
    )
    parser.add_argument("--early-gpu-claim", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.endpoint_file.exists():
        raise FileExistsError(args.endpoint_file)
    claim = claim_jax_device() if args.early_gpu_claim else None
    load_started = time.monotonic()
    policy, checkpoint_manifest = load_robotwin_pi05_policy(
        checkpoint=args.checkpoint,
        expected_stage=args.expected_stage,
        expected_step=args.expected_step,
        base_checkpoint=args.base_checkpoint,
        dataset=args.dataset,
        model_seed=args.model_seed,
        num_steps=args.num_denoising_steps,
        inference_profile=args.inference_profile,
    )
    load_seconds = time.monotonic() - load_started
    request_count = 0
    failures = 0
    normalization_statistics = json.loads(
        (args.dataset / "normalization_q01_q99.json").read_text(encoding="utf-8")
    )
    action_support = {"values": 0, "below_q01": 0, "above_q99": 0, "outside": 0}
    started = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        host, port = listener.getsockname()
        _atomic_json(
            args.endpoint_file,
            {
                "schema_version": 1,
                "status": "READY",
                "host": host,
                "port": port,
                "checkpoint": str(args.checkpoint.resolve()),
                "expected_step": args.expected_step,
                "task": json.loads((args.dataset / 'manifest.json').read_text())['task'],
                "inference_profile": args.inference_profile,
                "effective_interaction": effective_interaction(policy.mode),
                "load_seconds": load_seconds,
            },
        )
        with listener.accept()[0] as connection:
            connection.settimeout(1200)
            while args.max_requests <= 0 or request_count < args.max_requests:
                try:
                    request = receive_message(connection)
                except ConnectionError:
                    break
                try:
                    response = policy.infer(request)
                    current_support = action_support_summary(response["actions_r6"], normalization_statistics)
                    for key in action_support:
                        action_support[key] += current_support[key]
                except Exception as exc:
                    failures += 1
                    response = {
                        "schema_version": np.asarray(1, dtype=np.int32),
                        "status": np.asarray("error"),
                        "request_id": np.asarray(str(request.get("request_id", "unknown"))),
                        "error": np.asarray(f"{type(exc).__name__}: {exc}"),
                    }
                    traceback.print_exc()
                send_message(connection, response)
                request_count += 1
    _atomic_json(
        args.audit_file,
        {
            "schema_version": 1,
            "status": "PASS" if failures == 0 else "FAIL",
            "checkpoint_manifest": checkpoint_manifest,
            "requests": request_count,
            "failures": failures,
            "load_seconds": load_seconds,
            "service_seconds": time.monotonic() - started,
            "early_gpu_claim": args.early_gpu_claim,
            "inference_profile": args.inference_profile,
            "effective_interaction": effective_interaction(policy.mode),
            "action_support": {
                **action_support,
                "outside_fraction": action_support["outside"] / max(action_support["values"], 1),
            },
        },
    )
    del claim
    if failures:
        raise RuntimeError(f"policy server observed {failures} failed requests")


if __name__ == "__main__":
    main()
