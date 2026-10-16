#!/usr/bin/env python3
"""Generate and audit pinned q01/q99 statistics for PI0.5 Native V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EPSILON = 1e-6


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    q01 = np.quantile(values, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(values, 0.99, axis=0).astype(np.float32)
    if q01.shape != (10,) or q99.shape != (10,):
        raise ValueError("expected ten local dimensions")
    if not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q99 <= q01):
        raise ValueError("invalid q01/q99 statistics")
    return {"q01": q01.tolist(), "q99": q99.tolist(), "count": int(values.shape[0])}


def _audit(values: np.ndarray, stats: dict[str, list[float]]) -> dict:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    normalized = (values - q01) / (q99 - q01 + EPSILON) * 2.0 - 1.0
    restored = (normalized + 1.0) / 2.0 * (q99 - q01 + EPSILON) + q01
    return {
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "outside_minus1_plus1_percent": (
            np.mean((normalized < -1.0) | (normalized > 1.0), axis=0) * 100.0
        ).tolist(),
        "all_finite": bool(np.isfinite(normalized).all() and np.isfinite(restored).all()),
        "roundtrip_max_abs_error": float(np.max(np.abs(restored - values))),
    }


def main() -> None:
    args = _args()
    manifest_path = args.dataset / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    states = []
    actions = []
    for episode in manifest["episodes"]:
        path = args.dataset / episode["path"]
        if _sha256(path) != episode["sha256"]:
            raise ValueError(f"episode hash mismatch: {path}")
        with np.load(path) as archive:
            states.append(np.asarray(archive["proprioception"], dtype=np.float32))
            actions.append(np.asarray(archive["action"], dtype=np.float32))
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 20:
        raise ValueError(f"unexpected state/action shapes: {state.shape}, {action.shape}")

    groups = {
        "left": {"state": state[:, :10], "action": action[:, :10]},
        "right": {"state": state[:, 10:], "action": action[:, 10:]},
    }
    stats = {
        side: {name: _stats(values) for name, values in fields.items()}
        for side, fields in groups.items()
    }
    output = {
        "schema_version": 2,
        "normalization": "q01_q99",
        "epsilon": EPSILON,
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        **stats,
    }
    audit = {
        "schema_version": 1,
        "normalization_schema_version": 2,
        "dataset_manifest_sha256": output["dataset_manifest_sha256"],
        "episode_count": len(manifest["episodes"]),
        "step_count": int(state.shape[0]),
        "groups": {
            side: {
                name: _audit(values, stats[side][name])
                for name, values in fields.items()
            }
            for side, fields in groups.items()
        },
    }
    if not all(
        field["all_finite"] and field["roundtrip_max_abs_error"] <= 1e-5
        for fields in audit["groups"].values()
        for field in fields.values()
    ):
        raise ValueError("quantile normalization audit failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "audit_output": str(args.audit_output),
        "manifest_sha256": output["dataset_manifest_sha256"],
        "steps": int(state.shape[0]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
