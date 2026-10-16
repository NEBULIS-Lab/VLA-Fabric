"""Frozen paired-evaluation ranges and result auditing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class EvaluationProtocol:
    validation_ids: tuple[int, ...]
    fresh_ids: tuple[int, ...]

    @classmethod
    def standard(cls) -> "EvaluationProtocol":
        return cls(tuple(range(0, 200)), tuple(range(200, 400)))

    def ids(self, split: str) -> tuple[int, ...]:
        if split == "validation":
            return self.validation_ids
        if split == "fresh":
            return self.fresh_ids
        raise ValueError("split must be 'validation' or 'fresh'")


def checkpoint_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    files = (path,) if path.is_file() else tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    digest = hashlib.sha256()
    for file in files:
        relative = file.name if path.is_file() else file.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def freeze_evaluation(
    output_dir: str | Path,
    *,
    split: str,
    checkpoint: str | Path,
    model_name: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = EvaluationProtocol.standard().ids(split)
    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "split": split,
        "condition_ids": list(ids),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "pair_environment_seed": True,
        "pair_initial_state": True,
        "pair_flow_noise": True,
    }
    path = output_dir / "evaluation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def audit_rollout_rows(rows: Iterable[Mapping[str, object]], *, expected_ids: tuple[int, ...]) -> None:
    ids = [int(row["condition_id"]) for row in rows]
    duplicates = sorted(condition for condition, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate condition IDs: {duplicates}")
    missing = sorted(set(expected_ids) - set(ids))
    if missing:
        raise ValueError(f"missing condition IDs: {missing}")
    unexpected = sorted(set(ids) - set(expected_ids))
    if unexpected:
        raise ValueError(f"unexpected condition IDs: {unexpected}")
