"""Immutable data and scheduling contracts for shared bimanual training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np


_MANDATORY_TASKS = (
    "aloha_handover_box",
    "aloha_shoes_table",
    "robotwin_handover_mic",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _resolve(path: object, *, parent: Path) -> Path:
    candidate = Path(str(path)).expanduser()
    return candidate if candidate.is_absolute() else (parent / candidate).resolve()


@dataclass(frozen=True)
class SharedTaskSpec:
    name: str
    dataset: Path
    dataset_manifest_sha256: str
    normalizer: Path
    normalizer_sha256: str
    evaluation_manifest: Path | None
    evaluation_manifest_sha256: str | None
    action_dim: int
    state_dim: int
    camera_keys: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, object], *, parent: Path) -> "SharedTaskSpec":
        evaluation = value.get("evaluation_manifest")
        evaluation_sha = value.get("evaluation_manifest_sha256")
        if (evaluation is None) != (evaluation_sha is None):
            raise ValueError("evaluation manifest and hash must be provided together")
        return cls(
            name=str(value["name"]),
            dataset=_resolve(value["dataset"], parent=parent),
            dataset_manifest_sha256=_require_sha256(
                value["dataset_manifest_sha256"], field="dataset_manifest_sha256"
            ),
            normalizer=_resolve(value["normalizer"], parent=parent),
            normalizer_sha256=_require_sha256(
                value["normalizer_sha256"], field="normalizer_sha256"
            ),
            evaluation_manifest=(
                None if evaluation is None else _resolve(evaluation, parent=parent)
            ),
            evaluation_manifest_sha256=(
                None
                if evaluation_sha is None
                else _require_sha256(
                    evaluation_sha, field="evaluation_manifest_sha256"
                )
            ),
            action_dim=int(value["action_dim"]),
            state_dim=int(value["state_dim"]),
            camera_keys=tuple(str(item) for item in value["camera_keys"]),
        )

    def contract_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dataset": str(self.dataset),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "normalizer": str(self.normalizer),
            "normalizer_sha256": self.normalizer_sha256,
            "evaluation_manifest": (
                None if self.evaluation_manifest is None else str(self.evaluation_manifest)
            ),
            "evaluation_manifest_sha256": self.evaluation_manifest_sha256,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "camera_keys": list(self.camera_keys),
        }


@dataclass(frozen=True)
class SharedMixtureSpec:
    schema_version: int
    release_id: str
    condition: Literal["full", "independent"]
    tasks: tuple[SharedTaskSpec, ...]
    mixture_seed: int
    model_seed: int
    training_seed: int
    optimizer: dict[str, object]
    optimizer_steps: int | None
    checkpoint_steps: tuple[int, ...]
    formal_training_authorized: bool
    writable_paths: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path) -> "SharedMixtureSpec":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported shared-mixture schema")
        condition = str(payload.get("condition"))
        if condition not in ("full", "independent"):
            raise ValueError("condition must be full or independent")
        tasks = tuple(
            SharedTaskSpec.from_dict(item, parent=path.parent)
            for item in payload.get("tasks", ())
        )
        names = tuple(task.name for task in tasks)
        if len(set(names)) != len(names):
            raise ValueError("duplicate shared task name")
        if names[: len(_MANDATORY_TASKS)] != _MANDATORY_TASKS:
            raise ValueError("mandatory task order must be Box, Shoes, then Mic")
        if len(tasks) not in (3, 4):
            raise ValueError("shared training requires three or four tasks")
        action_dims = {task.action_dim for task in tasks}
        state_dims = {task.state_dim for task in tasks}
        camera_layouts = {task.camera_keys for task in tasks}
        if action_dims != {20}:
            raise ValueError("action_dim must be 20 for every task")
        if state_dims != {20}:
            raise ValueError("state_dim must be 20 for every task")
        if len(camera_layouts) != 1:
            raise ValueError("camera_keys must match across tasks")
        optimizer_steps = payload.get("optimizer_steps")
        if optimizer_steps is not None:
            optimizer_steps = int(optimizer_steps)
            if optimizer_steps <= 0:
                raise ValueError("optimizer_steps must be positive")
        authorized = bool(payload.get("formal_training_authorized", False))
        if authorized and optimizer_steps is None:
            raise ValueError("authorized training requires optimizer_steps")
        checkpoint_steps = tuple(int(step) for step in payload.get("checkpoint_steps", ()))
        if any(step <= 0 for step in checkpoint_steps):
            raise ValueError("checkpoint_steps must be positive")
        if optimizer_steps is not None and any(step > optimizer_steps for step in checkpoint_steps):
            raise ValueError("checkpoint_steps cannot exceed optimizer_steps")
        optimizer = dict(payload["optimizer"])
        if authorized:
            required_optimizer = {
                "learning_rate",
                "final_learning_rate",
                "warmup_steps",
                "batch_size",
                "device_count",
                "gradient_accumulation",
            }
            if not checkpoint_steps or checkpoint_steps[-1] != optimizer_steps:
                raise ValueError("authorized training requires a checkpoint at the final optimizer step")
            if not required_optimizer.issubset(optimizer):
                raise ValueError("authorized training requires a complete optimizer contract")
        writable_paths = {
            str(key): str(value)
            for key, value in dict(payload.get("writable_paths", {})).items()
        }
        return cls(
            schema_version=1,
            release_id=str(payload["release_id"]),
            condition=condition,
            tasks=tasks,
            mixture_seed=int(payload["mixture_seed"]),
            model_seed=int(payload["model_seed"]),
            training_seed=int(payload["training_seed"]),
            optimizer=optimizer,
            optimizer_steps=optimizer_steps,
            checkpoint_steps=checkpoint_steps,
            formal_training_authorized=authorized,
            writable_paths=writable_paths,
        )

    def validate_files(self) -> None:
        for task in self.tasks:
            dataset_manifest = task.dataset / "manifest.json"
            if not dataset_manifest.is_file():
                raise FileNotFoundError(f"missing dataset manifest for {task.name}")
            if _file_sha256(dataset_manifest) != task.dataset_manifest_sha256:
                raise ValueError(f"dataset manifest hash mismatch for {task.name}")
            if not task.normalizer.is_file():
                raise FileNotFoundError(f"missing normalizer for {task.name}")
            if _file_sha256(task.normalizer) != task.normalizer_sha256:
                raise ValueError(f"normalizer file hash mismatch for {task.name}")
            normalizer = json.loads(task.normalizer.read_text(encoding="utf-8"))
            if normalizer.get("dataset_manifest_sha256") != task.dataset_manifest_sha256:
                raise ValueError(f"normalizer does not match dataset for {task.name}")
            if task.evaluation_manifest is not None:
                if not task.evaluation_manifest.is_file():
                    raise FileNotFoundError(f"missing evaluation manifest for {task.name}")
                if _file_sha256(task.evaluation_manifest) != task.evaluation_manifest_sha256:
                    raise ValueError(f"evaluation manifest hash mismatch for {task.name}")

    def contract_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "condition": self.condition,
            "tasks": [task.contract_dict() for task in self.tasks],
            "mixture_seed": self.mixture_seed,
            "model_seed": self.model_seed,
            "training_seed": self.training_seed,
            "optimizer": self.optimizer,
            "optimizer_steps": self.optimizer_steps,
            "checkpoint_steps": list(self.checkpoint_steps),
            "formal_training_authorized": self.formal_training_authorized,
            "writable_paths": self.writable_paths,
        }


def _matched_contract(spec: SharedMixtureSpec) -> dict[str, object]:
    contract = spec.contract_dict()
    for field in ("condition", "release_id", "writable_paths"):
        contract.pop(field)
    return contract


def compare_matched_specs(full: SharedMixtureSpec, independent: SharedMixtureSpec) -> None:
    if full.condition != "full" or independent.condition != "independent":
        raise ValueError("matched specs require Full then Independent")
    if _matched_contract(full) != _matched_contract(independent):
        raise ValueError("Full and Independent matched contracts differ")


def task_index_for_step(task_count: int, *, seed: int, step: int) -> int:
    """Select one task from a deterministic once-per-block permutation."""

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if step < 0:
        raise ValueError("step must be nonnegative")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    block, offset = divmod(step, task_count)
    generator = np.random.default_rng(np.random.SeedSequence([seed, block]))
    return int(generator.permutation(task_count)[offset])


def task_counts_through_step(
    task_names: tuple[str, ...], *, seed: int, completed_steps: int
) -> dict[str, int]:
    """Count scheduled updates in the half-open interval [0, completed_steps)."""

    if not task_names:
        raise ValueError("task_names cannot be empty")
    if len(set(task_names)) != len(task_names):
        raise ValueError("task_names must be unique")
    if completed_steps < 0:
        raise ValueError("completed_steps must be nonnegative")
    task_count = len(task_names)
    complete_blocks, remainder = divmod(completed_steps, task_count)
    counts = {name: complete_blocks for name in task_names}
    start = complete_blocks * task_count
    for step in range(start, start + remainder):
        index = task_index_for_step(task_count, seed=seed, step=step)
        counts[task_names[index]] += 1
    return counts
