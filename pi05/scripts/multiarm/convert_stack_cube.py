#!/usr/bin/env python3
"""Build immutable Stack Cube indexes and role-specific quantile statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from pi05_fabric.data.stack_cube_multiagent import compute_dataset_statistics
from pi05_fabric.data.stack_cube_multiagent import validate_stack_cube_h5


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expected-trajectories", type=int, default=150)
args = parser.parse_args()

audit = validate_stack_cube_h5(
    args.source, expected_trajectories=args.expected_trajectories
)
statistics = compute_dataset_statistics(args.source)
args.output.mkdir(parents=True, exist_ok=True)

arrays = {}
for role, role_stats in statistics.items():
    for kind, stats in role_stats.items():
        arrays[f"{role}_{kind}_q01"] = stats.q01
        arrays[f"{role}_{kind}_q99"] = stats.q99
np.savez(args.output / "role_quantiles.npz", **arrays)

with h5py.File(args.source, "r") as handle:
    trajectories = [
        {
            "name": name,
            "steps": int(handle[f"{name}/actions/panda-0"].shape[0]),
        }
        for name in sorted(handle.keys())
    ]

digest = hashlib.sha256()
with args.source.open("rb") as source_file:
    for chunk in iter(lambda: source_file.read(16 * 1024 * 1024), b""):
        digest.update(chunk)

manifest = {
    **audit,
    "source": str(args.source.resolve()),
    "source_sha256": digest.hexdigest(),
    "prediction_horizon": 50,
    "execution_horizon": 25,
    "flow_steps": 10,
    "normalization": "q01_q99_per_role",
    "trajectories": trajectories,
    "sample_count": sum(item["steps"] for item in trajectories),
}
temporary = args.output / "dataset_manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary.replace(args.output / "dataset_manifest.json")
print(json.dumps(manifest, indent=2, sort_keys=True))
