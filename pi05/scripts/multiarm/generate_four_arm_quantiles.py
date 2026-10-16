#!/usr/bin/env python3
"""Generate and roundtrip-audit role-specific q01/q99 statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from pi05_fabric.data.four_arm_tasks import FourArmQuantileNormalization
from pi05_fabric.data.four_arm_tasks import compute_four_arm_quantiles
from pi05_fabric.data.four_arm_tasks import normalize_quantile
from pi05_fabric.data.four_arm_tasks import unnormalize_quantile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    compute_four_arm_quantiles(args.dataset, args.output)
    normalization = FourArmQuantileNormalization.from_npz(args.output)
    max_error = 0.0
    with h5py.File(args.dataset, "r") as handle:
        trajectories = sorted(handle.keys())
        for role in range(4):
            stats = normalization.for_role(role).action
            for name in trajectories:
                actions = np.asarray(handle[f"{name}/actions/role_{role}"], dtype=np.float32)
                restored = unnormalize_quantile(normalize_quantile(actions, stats), stats)
                max_error = max(max_error, float(np.max(np.abs(restored - actions))))
    if not np.isfinite(max_error) or max_error > 1e-4:
        raise ValueError(f"action normalization roundtrip error is {max_error}")
    report = {
        "dataset": str(args.dataset.resolve()),
        "normalization": str(args.output.resolve()),
        "roles": 4,
        "max_action_roundtrip_error": max_error,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
