"""Small, testable helpers for resumable Fresh Paired-200 rolling dispatch."""

from __future__ import annotations

from typing import Callable, Sequence

from .fresh_paired import FrozenEvalJob


def job_key(job: FrozenEvalJob) -> str:
    return f"{job.task}/{job.run_id}/{job.rollout_offset}"


def split_completed_jobs(
    jobs: Sequence[FrozenEvalJob],
    *,
    is_complete: Callable[[FrozenEvalJob], bool],
) -> tuple[list[FrozenEvalJob], list[FrozenEvalJob]]:
    pending: list[FrozenEvalJob] = []
    completed: list[FrozenEvalJob] = []
    for job in jobs:
        (completed if is_complete(job) else pending).append(job)
    return pending, completed

