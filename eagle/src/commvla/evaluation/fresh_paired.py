"""Protocol helpers for the two-task Fresh Paired-200 campaign."""

from __future__ import annotations

import json
import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TypeVar


FRESH_IDS = tuple(range(200, 400))
COMM_GPU_PAIRS = ((0, 1), (2, 3), (4, 5))
TASKS = ("aloha_handover_box", "aloha_shoes_table")
TASK_RUN_IDS = {
    "aloha_handover_box": (
        "D-I0", "D-I1", "D-I2", "S-I2", "D-I3", "S-I3", "D-I3-RAW", "S-I3-RAW",
        "D-I4-AVG", "S-I4-AVG", "D-I4-RAW", "S-I4-RAW", "D-I5-AVG", "S-I5-AVG",
        "D-I5-RAW", "S-I5-RAW",
    ),
    "aloha_shoes_table": (
        "SHOES-D-I0", "SHOES-D-I1", "SHOES-D-I2", "SHOES-S-I2",
        "SHOES-D-I3-AVG", "SHOES-S-I3-AVG", "SHOES-D-I3-RAW", "SHOES-S-I3-RAW",
        "SHOES-D-I4-AVG", "SHOES-S-I4-AVG", "SHOES-D-I4-RAW", "SHOES-S-I4-RAW",
        "SHOES-D-I5-AVG", "SHOES-S-I5-AVG", "SHOES-D-I5-RAW", "SHOES-S-I5-RAW",
    ),
}


def validate_devices(devices: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(device) for device in devices)
    if any(device in (6, 7) for device in normalized):
        raise ValueError("GPU 6/7 are reserved and cannot be scheduled")
    if not normalized or any(device < 0 or device > 5 for device in normalized):
        raise ValueError("Fresh Paired-200 jobs must use GPU 0--5")
    return normalized


@dataclass(frozen=True)
class WaveJob:
    run_id: str
    task: str
    checkpoint: str
    rollout_offset: int
    trials: int
    devices: tuple[int, int]
    port: int


@dataclass(frozen=True)
class FrozenEvalJob:
    kind: str
    run_id: str
    task: str
    checkpoint: str
    rollout_offset: int
    trials: int = 100


JobT = TypeVar("JobT")
SlotT = TypeVar("SlotT")
ResultT = TypeVar("ResultT")


def build_safe_comm_waves(
    models: Sequence[Mapping[str, object]], port_base: int
) -> list[list[WaveJob]]:
    if len(models) != 16:
        raise ValueError("A task campaign requires exactly 16 CommVLA models")
    jobs: list[WaveJob] = []
    for model in models:
        if model.get("kind") != "commvla":
            raise ValueError("Only CommVLA entries may be placed in CommVLA waves")
        for rollout_offset in (200, 300):
            job_index = len(jobs)
            jobs.append(
                WaveJob(
                    run_id=str(model["run_id"]),
                    task=str(model["task"]),
                    checkpoint=str(model["checkpoint"]),
                    rollout_offset=rollout_offset,
                    trials=100,
                    devices=COMM_GPU_PAIRS[job_index % len(COMM_GPU_PAIRS)],
                    port=port_base + job_index,
                )
            )
    return [jobs[index : index + 3] for index in range(0, len(jobs), 3)]


def build_tail_twin_jobs(model: Mapping[str, object]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if model.get("kind") != "twinvla":
        raise ValueError("Tail single-GPU jobs require a TwinVLA entry")
    return ((200, (4,)), (300, (5,)))


def build_rolling_jobs(
    entries: Sequence[Mapping[str, object]],
) -> tuple[list[FrozenEvalJob], list[FrozenEvalJob]]:
    comm: list[FrozenEvalJob] = []
    twin: list[FrozenEvalJob] = []
    for entry in entries:
        kind = str(entry.get("kind"))
        if kind not in ("commvla", "twinvla"):
            raise ValueError(f"Unsupported frozen entry kind: {kind}")
        target = comm if kind == "commvla" else twin
        for rollout_offset in (200, 300):
            target.append(
                FrozenEvalJob(
                    kind=kind,
                    run_id=str(entry["run_id"]),
                    task=str(entry["task"]),
                    checkpoint=str(entry["checkpoint"]),
                    rollout_offset=rollout_offset,
                )
            )
    return comm, twin


def run_rolling_workers(
    jobs: Sequence[JobT],
    slots: Sequence[SlotT],
    run_job: Callable[[JobT, SlotT], ResultT],
) -> list[ResultT]:
    pending: queue.Queue[JobT] = queue.Queue()
    for job in jobs:
        pending.put(job)
    stopped = threading.Event()
    completed: list[ResultT] = []
    completed_lock = threading.Lock()

    def consume(slot: SlotT) -> None:
        while not stopped.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_job(job, slot)
            except BaseException:
                stopped.set()
                raise
            finally:
                pending.task_done()
            with completed_lock:
                completed.append(result)

    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(consume, slot) for slot in slots]
        for future in futures:
            future.result()
    return completed


def selection_to_entry(selection_path: Path, task: str, *, checkpoint_sha256: str) -> dict[str, object]:
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    if data.get("cleanup_performed") is not True:
        raise ValueError(f"Selection cleanup is not complete: {selection_path}")
    selected_tag = str(data.get("recommended_best", ""))
    candidates = [candidate for candidate in data.get("candidate_results", []) if candidate.get("tag") == selected_tag]
    if len(candidates) != 1:
        raise ValueError(f"Selection does not resolve exactly one best checkpoint: {selection_path}")
    checkpoint = Path(str(candidates[0]["checkpoint"]))
    if not checkpoint.is_dir():
        raise ValueError(f"Selected checkpoint is missing: {checkpoint}")
    return {
        "kind": "commvla",
        "run_id": str(data["run_id"]),
        "task": task,
        "checkpoint": str(checkpoint),
        "selection_file": str(selection_path),
        "selected_tag": selected_tag,
        "checkpoint_sha256": checkpoint_sha256,
    }


def validate_frozen_manifest(manifest: Mapping[str, object], *, check_paths: bool = True) -> None:
    if manifest.get("fresh_ids") != [200, 399]:
        raise ValueError("Fresh ID range must be exactly 200--399")
    if int(manifest.get("seed", -1)) != 1501:
        raise ValueError("Fresh campaign seed must be 1501")
    entries = list(manifest.get("entries", []))
    seen: set[tuple[str, str]] = set()
    for task in TASKS:
        task_entries = [entry for entry in entries if entry.get("task") == task]
        comm = [entry for entry in task_entries if entry.get("kind") == "commvla"]
        twin = [entry for entry in task_entries if entry.get("kind") == "twinvla"]
        if len(comm) != 16:
            raise ValueError(f"{task} must contain 16 CommVLA entries")
        if len(twin) != 1:
            raise ValueError(f"{task} must contain one TwinVLA entry")
        for entry in task_entries:
            key = (task, str(entry.get("run_id")))
            if key in seen:
                raise ValueError(f"Duplicate model entry: {key}")
            seen.add(key)
            digest = str(entry.get("checkpoint_sha256", ""))
            if len(digest) != 64:
                raise ValueError(f"Missing checkpoint SHA256 for {key}")
            if check_paths and not Path(str(entry.get("checkpoint", ""))).is_dir():
                raise ValueError(f"Checkpoint directory does not exist for {key}")
    if len(entries) != 34:
        raise ValueError("Frozen manifest must contain 32 CommVLA and two TwinVLA entries")


def validate_comm_summary(
    summary_path: Path,
    model: Mapping[str, object],
    *,
    rollout_offset: int,
    trials: int = 100,
) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_ids = list(range(rollout_offset, rollout_offset + trials))
    if data.get("rollout_ids") != expected_ids:
        raise ValueError(f"Unexpected rollout IDs in {summary_path}")
    checks = {
        "checkpoint": str(model["checkpoint"]),
        "task_name": str(model["task"]),
        "trials": trials,
        "plan_seed_offset": 0,
        "execution_mode": "chunk",
        "action_len": 20,
        "dtype": "bfloat16",
        "cfg": 1.1,
        "num_denoising_steps": 10,
    }
    for key, expected in checks.items():
        if data.get(key) != expected:
            raise ValueError(f"Unexpected {key} in {summary_path}: {data.get(key)!r} != {expected!r}")
    rows = data.get("rows", [])
    row_ids = [int(row["rollout_id"]) for row in rows]
    if row_ids != expected_ids or len(set(row_ids)) != trials:
        raise ValueError(f"Unexpected rollout IDs in rows for {summary_path}")
    if any(int(row.get("finite_all", 0)) != 1 for row in rows):
        raise ValueError(f"Non-finite model output in {summary_path}")
    return data


def validate_twin_summary(
    summary_path: Path,
    model: Mapping[str, object],
    *,
    rollout_offset: int,
    trials: int = 100,
) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_ids = list(range(rollout_offset, rollout_offset + trials))
    checks = {
        "kind": "twinvla",
        "checkpoint": str(model["checkpoint"]),
        "task_name": str(model["task"]),
        "trials": trials,
        "rollout_ids": expected_ids,
        "seed": 1501,
        "plan_seed_offset": 0,
        "cfg": 1.1,
        "num_denoising_steps": 10,
        "dtype": "bfloat16",
        "execution_mode": "chunk",
        "action_len": 20,
    }
    for key, expected in checks.items():
        if data.get(key) != expected:
            raise ValueError(f"Unexpected {key} in {summary_path}: {data.get(key)!r} != {expected!r}")
    rows = data.get("rows", [])
    if [int(row["rollout_id"]) for row in rows] != expected_ids:
        raise ValueError(f"Unexpected rollout IDs in rows for {summary_path}")
    if any(int(row.get("finite_all", 0)) != 1 for row in rows):
        raise ValueError(f"Non-finite TwinVLA output in {summary_path}")
    return data


def cleanup_smoke_workspace(workspace: Path, audit_path: Path, audit: Mapping[str, object]) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(dict(audit), indent=2, sort_keys=True), encoding="utf-8")


def exact_ids(rows: Iterable[Mapping[str, object]]) -> tuple[int, ...]:
    return tuple(int(row["rollout_id"]) for row in rows)
