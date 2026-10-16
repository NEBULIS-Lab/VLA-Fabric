"""Immutable condition and rollout contracts for RoboTwin2 evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable, Mapping

from pi05_fabric.evaluation.robotwin2_task_diagnostics import PHASE_FIELDS
from pi05_fabric.evaluation.robotwin2_task_diagnostics import canonical_task_name
from pi05_fabric.evaluation.robotwin2_task_diagnostics import task_diagnostics


@dataclass(frozen=True)
class RobotwinCondition:
    condition_id: int
    env_seed: int
    instruction: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_condition_csv(
    path: str | Path, *, expected_ids: tuple[int, ...]
) -> tuple[RobotwinCondition, ...]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    conditions = tuple(
        RobotwinCondition(
            condition_id=int(row.get("condition_id") or row.get("rollout_id", "")),
            env_seed=int(row["env_seed"]),
            instruction=str(row["instruction"]).strip(),
        )
        for row in rows
    )
    ids = tuple(condition.condition_id for condition in conditions)
    if ids != expected_ids:
        missing = sorted(set(expected_ids) - set(ids))
        unexpected = sorted(set(ids) - set(expected_ids))
        raise ValueError(
            f"condition IDs do not match expected ordered range; missing={missing}, "
            f"unexpected={unexpected}"
        )
    seeds = tuple(condition.env_seed for condition in conditions)
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("condition environment seeds must be non-negative and unique")
    if any(not condition.instruction for condition in conditions):
        raise ValueError("condition instructions must be non-empty")
    return conditions


def write_condition_csv(path: str | Path, conditions: Iterable[RobotwinCondition]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("condition_id", "env_seed", "instruction")
        )
        writer.writeheader()
        for condition in conditions:
            writer.writerow(
                {
                    "condition_id": condition.condition_id,
                    "env_seed": condition.env_seed,
                    "instruction": condition.instruction,
                }
            )


def merge_condition_csvs(
    sources: Iterable[str | Path],
    *,
    output: str | Path,
    expected_ids: tuple[int, ...],
) -> tuple[RobotwinCondition, ...]:
    merged: list[RobotwinCondition] = []
    for source in sources:
        with Path(source).open(newline="", encoding="utf-8") as stream:
            rows = tuple(csv.DictReader(stream))
        for row in rows:
            merged.append(
                RobotwinCondition(
                    int(row.get("condition_id") or row.get("rollout_id", "")),
                    int(row["env_seed"]),
                    str(row["instruction"]).strip(),
                )
            )
    write_condition_csv(output, merged)
    return load_condition_csv(output, expected_ids=expected_ids)


def freeze_condition_split(
    source: str | Path,
    *,
    output_dir: str | Path,
    split: str,
    expected_ids: tuple[int, ...],
    expert_validated: bool,
    task_name: str = "robotwin_handover_block",
) -> Path:
    task = canonical_task_name(task_name)
    source = Path(source).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"condition split already exists: {output_dir}")
    conditions = load_condition_csv(source, expected_ids=expected_ids)
    output_dir.mkdir(parents=True)
    frozen = output_dir / "conditions.csv"
    write_condition_csv(frozen, conditions)
    manifest = {
        "schema_version": 1,
        "task": task,
        "split": split,
        "condition_count": len(conditions),
        "first_condition_id": conditions[0].condition_id,
        "last_condition_id": conditions[-1].condition_id,
        "expert_validated": bool(expert_validated),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "frozen_sha256": file_sha256(frozen),
        "pair_environment_seed": True,
        "pair_initial_state": True,
        "pair_flow_noise": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


_TIMING_FIELDS = (
    "mean_plan_seconds",
    "p95_plan_seconds",
    "episode_seconds",
)


def audit_robotwin_rollout_rows(
    rows: Iterable[Mapping[str, object]], *, conditions: tuple[RobotwinCondition, ...]
) -> None:
    rows = tuple(rows)
    expected = {condition.condition_id: condition for condition in conditions}
    observed_ids = [int(row["condition_id"]) for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("duplicate condition IDs in rollout rows")
    missing = sorted(set(expected) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(expected))
    if missing:
        raise ValueError(f"missing condition IDs: {missing}")
    if unexpected:
        raise ValueError(f"unexpected condition IDs: {unexpected}")
    tasks = {
        canonical_task_name(str(row.get("task", "robotwin_handover_block")))
        for row in rows
    }
    if len(tasks) > 1:
        raise ValueError("mixed tasks in rollout rows")
    for row in rows:
        condition = expected[int(row["condition_id"])]
        if int(row["env_seed"]) != condition.env_seed:
            raise ValueError(f"environment seed mismatch for condition {condition.condition_id}")
        if int(row["steps"]) < 0 or int(row["planning_calls"]) < 0:
            raise ValueError("steps and planning_calls must be non-negative")
        if int(row["success"]) not in (0, 1) or int(row["timeout"]) not in (0, 1):
            raise ValueError("success and timeout must be binary")
        for field in _TIMING_FIELDS:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        task = canonical_task_name(str(row.get("task", "robotwin_handover_block")))
        status = row.get("phase_metrics_status")
        # Legacy Block rows need no new columns. New task adapters record diagnostic availability.
        if task != "robotwin_handover_block" or status is not None:
            expected_status = task_diagnostics(task).phase_metrics_status
            if status != expected_status:
                raise ValueError(f"phase_metrics_status must be {expected_status} for {task}")
            for field in PHASE_FIELDS:
                if field not in row:
                    raise ValueError(f"missing phase metric: {field}")
                if status == "not_collected":
                    if row[field] not in (None, ""):
                        raise ValueError(f"{field} must be empty when not_collected")
                elif str(row[field]) not in ("0", "1"):
                    raise ValueError(f"{field} must be binary when collected")
        official = row.get("official_step_limit")
        effective = row.get("effective_step_limit")
        if official not in (None, "") or effective not in (None, ""):
            if official in (None, "") or effective in (None, ""):
                raise ValueError("official and effective step limits must be recorded together")
            if not 0 < int(effective) <= int(official) or int(row["steps"]) > int(effective):
                raise ValueError("steps or effective step limit exceeds official horizon")
            if task == "robotwin_handover_mic" and int(effective) > 800:
                raise ValueError("Mic effective step limit exceeds 800-step cap")


def summarize_phase_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Preserve legacy Block counts and serialize uncollected diagnostics as null."""
    rows = tuple(rows)
    tasks = {
        canonical_task_name(str(row.get("task", "robotwin_handover_block")))
        for row in rows
    }
    if len(tasks) != 1:
        raise ValueError("phase summary requires non-empty rows from one task")
    task = tasks.pop()
    status = task_diagnostics(task).phase_metrics_status
    counts = {}
    for field in PHASE_FIELDS:
        if status == "not_collected":
            if any(row.get("phase_metrics_status") != status or row.get(field) not in (None, "") for row in rows):
                raise ValueError(f"{task} phase metrics must be explicitly not_collected")
            counts[field] = None
        else:
            counts[field] = sum(int(row[field]) for row in rows)
    return {"task": task, "phase_metrics_status": status, **counts}


def merge_rollout_shards(
    sources: Iterable[str | Path],
    *,
    output_dir: str | Path,
    conditions: tuple[RobotwinCondition, ...],
    purpose: str,
) -> dict[str, object]:
    """Audit and merge disjoint rollout shards into one deterministic result."""
    sources = tuple(Path(source) for source in sources)
    if not sources:
        raise ValueError("at least one rollout shard is required")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"merged rollout output already exists: {output_dir}")

    rows: list[dict[str, str]] = []
    fieldnames: tuple[str, ...] | None = None
    for source in sources:
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            current_fields = tuple(reader.fieldnames or ())
            if not current_fields:
                raise ValueError(f"rollout shard has no header: {source}")
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"rollout shard fields differ: {source}")
            rows.extend(dict(row) for row in reader)

    rows.sort(key=lambda row: int(row["condition_id"]))
    audit_robotwin_rollout_rows(rows, conditions=conditions)
    output_dir.mkdir(parents=True)
    with (output_dir / "rollouts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = sum(bool(str(row["error"]).strip()) for row in rows)
    summary = {
        "schema_version": 1,
        "status": "PASS" if errors == 0 else "FAIL",
        "purpose": purpose,
        "successes": sum(int(row["success"]) for row in rows),
        "trials": len(rows),
        "timeouts": sum(int(row["timeout"]) for row in rows),
        "errors": errors,
        "mean_planning_calls": sum(int(row["planning_calls"]) for row in rows)
        / len(rows),
        **summarize_phase_metrics(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    merge_manifest = {
        "schema_version": 1,
        "task": summary["task"],
        "phase_metrics_status": summary["phase_metrics_status"],
        "sources": [
            {"path": str(source.resolve()), "sha256": file_sha256(source)}
            for source in sources
        ],
        "condition_ids": [condition.condition_id for condition in conditions],
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(merge_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
