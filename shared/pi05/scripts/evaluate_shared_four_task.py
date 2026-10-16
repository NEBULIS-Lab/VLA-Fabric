#!/usr/bin/env python3
"""Run one PI0.5 Shared task through a registered environment overlay."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from pi05_fabric.agents.dual_pi05 import SharedInferenceMode
from pi05_fabric.agents.dual_pi05 import shared_effective_channels
from pi05_fabric.evaluation.protocol import checkpoint_sha256
from pi05_fabric.evaluation.shared_four_task import SHARED_TASKS
from pi05_fabric.evaluation.shared_four_task import evaluate_shared_task


def _load_factory(entrypoint: str):
    try:
        module_name, function_name = entrypoint.split(":", 1)
    except ValueError as error:
        raise ValueError("runtime factory must use module:function syntax") from error
    return getattr(importlib.import_module(module_name), function_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=SHARED_TASKS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "all_off"), required=True)
    parser.add_argument("--runtime-factory", required=True, help="module:function returning (adapter, policy)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-step", type=int, help="Required for development checkpoint selection")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = SharedInferenceMode(args.mode)
    adapter, policy = _load_factory(args.runtime_factory)(args)
    summary = evaluate_shared_task(
        architecture="pi05",
        adapter=adapter,
        policy=policy,
        condition_manifest=json.loads(args.conditions.read_text(encoding="utf-8")),
        mode=mode.value,
        checkpoint_sha256=checkpoint_sha256(args.checkpoint),
        effective_channels=shared_effective_channels(mode),
        output_dir=args.output_dir,
        formal=not args.smoke,
        checkpoint_step=args.checkpoint_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
