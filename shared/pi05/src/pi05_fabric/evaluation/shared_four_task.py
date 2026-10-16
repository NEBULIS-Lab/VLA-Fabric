"""Architecture-neutral rollout contract for the four Shared tasks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


SHARED_TASKS = (
    "aloha_handover_box",
    "aloha_shoes_table",
    "robotwin_handover_mic",
    "robotwin_scan_object",
)
CHANNELS = ("common", "private_kv", "action")


@dataclass(frozen=True)
class SharedTaskAdapter:
    """Small boundary between the common audit loop and a task runtime."""

    name: str
    environment_protocol: str
    load_condition: Callable[[dict], object]
    make_observation: Callable[[object], object]
    step: Callable[[object, object], tuple[object, bool]]
    success: Callable[[object], bool]
    max_steps: int = 1000


def _validate_manifest(
    adapter: SharedTaskAdapter, condition_manifest: Mapping[str, object], *, formal: bool
) -> list[dict]:
    if condition_manifest.get("schema_version") != 1:
        raise ValueError("unsupported frozen-condition schema")
    if condition_manifest.get("task") != adapter.name:
        raise ValueError("condition manifest task does not match the adapter")
    if condition_manifest.get("environment_protocol_id") != adapter.environment_protocol:
        raise ValueError("condition manifest protocol does not match the adapter")
    conditions = list(condition_manifest.get("conditions", ()))
    expected_count = 200 if formal else 1
    if len(conditions) != expected_count:
        raise ValueError(f"{'formal' if formal else 'smoke'} evaluation requires exactly {expected_count} conditions")
    if int(condition_manifest.get("count", -1)) != len(conditions):
        raise ValueError("condition count field does not match its records")
    identifiers = [item.get("condition_id") for item in conditions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("condition manifest contains duplicate condition IDs")
    if any(not isinstance(identifier, int) for identifier in identifiers):
        raise ValueError("condition IDs must be integers")
    return sorted(conditions, key=lambda item: item["condition_id"])


def _failure_row(
    base: dict[str, object], error: BaseException, *, steps: int, planning_calls: int
) -> dict[str, object]:
    timeout = isinstance(error, TimeoutError)
    nonfinite = isinstance(error, FloatingPointError)
    return {
        **base,
        "success": 0,
        "steps": steps,
        "planning_calls": planning_calls,
        "timeout": int(timeout),
        "finite": int(not nonfinite),
        "error": str(error),
    }


def _array_leaves(value: object) -> list[np.ndarray]:
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _array_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _array_leaves(item)]
    return [np.asarray(value)]


def _run_condition(
    *,
    architecture: str,
    adapter: SharedTaskAdapter,
    policy: Callable,
    condition: dict,
    mode: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    condition_id = int(condition["condition_id"])
    base = {
        "architecture": architecture,
        "task": adapter.name,
        "mode": mode,
        "condition_id": condition_id,
        "checkpoint_sha256": checkpoint_sha256,
    }
    steps = 0
    planning_calls = 0
    try:
        state = adapter.load_condition(condition)
        done = False
        while not done and steps < adapter.max_steps:
            observation = adapter.make_observation(state)
            action = policy(
                observation,
                condition_id=condition_id,
                planning_call=planning_calls,
            )
            planning_calls += 1
            leaves = _array_leaves(action)
            if not all(np.isfinite(item).all() for item in leaves):
                raise FloatingPointError("policy produced a non-finite action")
            state, done = adapter.step(state, action)
            steps += 1
        if not done:
            raise TimeoutError(f"condition exceeded {adapter.max_steps} environment steps")
        return {
            **base,
            "success": int(adapter.success(state)),
            "steps": steps,
            "planning_calls": planning_calls,
            "timeout": 0,
            "finite": 1,
            "error": "",
        }
    except Exception as error:
        return _failure_row(base, error, steps=steps, planning_calls=planning_calls)


def evaluate_shared_task(
    *,
    architecture: str,
    adapter: SharedTaskAdapter,
    policy: Callable,
    condition_manifest: Mapping[str, object],
    mode: str,
    checkpoint_sha256: str,
    effective_channels: Mapping[str, bool],
    output_dir: Path,
    formal: bool,
    checkpoint_step: int | None = None,
) -> dict[str, object]:
    if adapter.name not in SHARED_TASKS:
        raise ValueError(f"unregistered Shared task: {adapter.name}")
    if mode not in ("full", "all_off"):
        raise ValueError("Shared inference mode must be full or all_off")
    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must be a SHA-256 digest")
    channels = {name: bool(effective_channels[name]) for name in CHANNELS}
    expected = mode == "full"
    if any(value != expected for value in channels.values()):
        raise ValueError("effective channels do not match the Shared inference mode")
    conditions = _validate_manifest(adapter, condition_manifest, formal=formal)
    split = condition_manifest.get("split")
    if split not in ("development", "fresh"):
        raise ValueError("condition split must be development or fresh")
    if split == "development" and (not isinstance(checkpoint_step, int) or checkpoint_step <= 0):
        raise ValueError("development evaluation requires a positive checkpoint_step")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "architecture": architecture,
        "task": adapter.name,
        "mode": mode,
        "formal": formal,
        "checkpoint_training_mode": "full",
        "checkpoint_sha256": checkpoint_sha256,
        "environment_protocol_id": adapter.environment_protocol,
        "effective_channels": channels,
        "split": split, "checkpoint_step": checkpoint_step,
        "expected_condition_ids": [item["condition_id"] for item in conditions],
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        _run_condition(
            architecture=architecture,
            adapter=adapter,
            policy=policy,
            condition=condition,
            mode=mode,
            checkpoint_sha256=checkpoint_sha256,
        )
        for condition in conditions
    ]
    fields = (
        "architecture", "task", "mode", "condition_id", "success", "steps",
        "planning_calls", "timeout", "finite", "error", "checkpoint_sha256",
    )
    with (output_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    observed = [int(row["condition_id"]) for row in rows]
    expected_ids = manifest["expected_condition_ids"]
    if observed != expected_ids:
        raise RuntimeError("rollout rows do not exactly cover the frozen condition IDs")
    summary = {
        **manifest,
        "status": "complete",
        "trials": len(rows),
        "attempts": len(rows), "condition_ids": observed,
        "successes": sum(int(row["success"]) for row in rows),
        "timeouts": sum(int(row["timeout"]) for row in rows),
        "nonfinite": sum(not int(row["finite"]) for row in rows),
        "non_finite": sum(not int(row["finite"]) for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
