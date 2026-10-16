"""Validation-200 checkpoint candidates, rollout auditing, and selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from pi05_fabric.evaluation.protocol import EvaluationProtocol
from pi05_fabric.evaluation.protocol import audit_rollout_rows


_CANDIDATE_STEPS = (10_000, 20_000, 30_000, 40_000)
_EXPECTED_STAGE = "i4_raw_full_stage2"
_NONPAPER_SMOKE_IDS = frozenset(range(9050, 9100))


@dataclass(frozen=True)
class CandidateSpec:
    step: int
    gpu: int
    checkpoint: Path
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class CandidateResult:
    step: int
    successes: int
    trials: int
    timeouts: int
    errors: int
    mean_success_steps: float


def build_candidate_specs(checkpoint_root: str | Path) -> tuple[CandidateSpec, ...]:
    checkpoint_root = Path(checkpoint_root)
    specs = []
    for gpu, step in enumerate(_CANDIDATE_STEPS):
        checkpoint = checkpoint_root / f"step_{step:08d}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"missing Stage2 checkpoint at step {step}: {checkpoint}")
        manifest_path = checkpoint / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("stage") != _EXPECTED_STAGE:
            raise ValueError(f"checkpoint {step} has unexpected stage: {manifest.get('stage')}")
        if int(manifest.get("step", -1)) != step:
            raise ValueError(f"checkpoint directory and manifest step disagree at {step}")
        specs.append(CandidateSpec(step, gpu, checkpoint, manifest))
    return tuple(specs)


def requested_condition_ids(value: str, *, allow_nonpaper_smoke: bool) -> tuple[int, ...]:
    if value == "validation":
        return EvaluationProtocol.standard().validation_ids
    if value == "fresh":
        return EvaluationProtocol.standard().fresh_ids
    if not allow_nonpaper_smoke:
        raise ValueError("explicit condition IDs are allowed only for non-paper smoke")
    ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("non-paper smoke condition IDs must be non-empty and unique")
    if not set(ids).issubset(_NONPAPER_SMOKE_IDS):
        raise ValueError("condition IDs are outside the reserved non-paper smoke range 9050-9099")
    return ids


def summarize_candidate_rows(
    *, step: int, rows: Iterable[Mapping[str, object]], expected_ids: tuple[int, ...] | None = None
) -> CandidateResult:
    rows = tuple(rows)
    audit_rollout_rows(
        rows, expected_ids=expected_ids or EvaluationProtocol.standard().validation_ids
    )
    if any(not bool(int(row["finite_all"])) for row in rows):
        raise ValueError("non-finite rollout output")
    successes = tuple(row for row in rows if bool(int(row["success"])))
    mean_success_steps = (
        sum(float(row["steps"]) for row in successes) / len(successes)
        if successes
        else float("inf")
    )
    return CandidateResult(
        step=step,
        successes=len(successes),
        trials=len(rows),
        timeouts=sum(int(row["timeout"]) for row in rows),
        errors=sum(bool(str(row["error"])) for row in rows),
        mean_success_steps=mean_success_steps,
    )


def select_best_candidate(candidates: Iterable[CandidateResult]) -> CandidateResult:
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one candidate result is required")
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.successes,
            candidate.timeouts + candidate.errors,
            candidate.mean_success_steps,
            candidate.step,
        ),
    )
