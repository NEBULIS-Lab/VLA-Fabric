#!/usr/bin/env python3
"""Convert one formal four-arm release into the PI0.5 task layout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from pi05_fabric.data.four_arm_tasks import convert_four_arm_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    report = asdict(convert_four_arm_release(args.source, args.output))
    report["source"] = str(report["source"])
    report["output"] = str(report["output"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
