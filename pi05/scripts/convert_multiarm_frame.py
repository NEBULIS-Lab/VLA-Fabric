#!/usr/bin/env python3
"""Convert and normalize one audited multi-arm frame release."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from pi05_fabric.data.multiarm_frame_tasks import compute_multiarm_quantiles
from pi05_fabric.data.multiarm_frame_tasks import convert_multiarm_release
from pi05_fabric.data.multiarm_frame_tasks import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantiles", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--agent-count", type=int, choices=(3, 4), required=True)
    args = parser.parse_args()

    report = asdict(
        convert_multiarm_release(
            args.source,
            args.output,
            agent_count=args.agent_count,
        )
    )
    compute_multiarm_quantiles(
        args.output,
        args.quantiles,
        expected_agent_count=args.agent_count,
    )
    report["source"] = str(report["source"])
    report["output"] = str(report["output"])
    report["output_sha256"] = sha256_file(args.output)
    report["quantiles"] = str(args.quantiles.resolve())
    report["quantiles_sha256"] = sha256_file(args.quantiles)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
