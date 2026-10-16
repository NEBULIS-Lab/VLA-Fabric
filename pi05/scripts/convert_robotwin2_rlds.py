#!/usr/bin/env python3
"""Convert and audit task-pinned RoboTwin2 RLDS data (Block by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_aloha_rlds import convert
from pi05_fabric.data.robotwin2_dataset import EXPECTED_TASK
from pi05_fabric.data.robotwin2_dataset import TASK_EXPECTED_STEPS
from pi05_fabric.data.robotwin2_dataset import audit_converted_dataset
from pi05_fabric.data.robotwin2_dataset import audit_source_dataset
from pi05_fabric.data.robotwin2_dataset import validate_source_metadata


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", choices=tuple(TASK_EXPECTED_STEPS), default=EXPECTED_TASK)
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    source = validate_source_metadata(args.dataset_dir, task=args.task_name)
    needs_source_audit = TASK_EXPECTED_STEPS[args.task_name] is None
    if needs_source_audit:
        source = audit_source_dataset(args.dataset_dir, task=args.task_name)
    convert(
        args.dataset_dir,
        args.output_dir,
        max_episodes=args.max_episodes,
        profile="robotwin2",
    )
    if needs_source_audit:
        source_path = args.output_dir / "robotwin2_source_audit.json"
        source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    audit = audit_converted_dataset(
        args.output_dir,
        require_full_dataset=args.max_episodes is None,
        task=args.task_name,
    )
    payload = {"source": source, "converted": audit}
    path = args.output_dir / "robotwin2_conversion_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
