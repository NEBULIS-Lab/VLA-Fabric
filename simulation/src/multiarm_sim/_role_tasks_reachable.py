"""Reviewed public contracts for short role-diverse multi-arm previews."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from multiarm_sim._role_tasks_definitions import (
    F1,
    F2,
    F3,
    F4,
    F5,
    FOUR_ARM_BASE_XY,
    GLOBAL_CAMERAS,
    TABLE_HEIGHT,
    THREE_ARM_BASE_XY,
    ObjectKeyframe,
    PreviewObjectSpec,
    RolePhase,
    RoleTaskDefinition,
    T1 as _T1_DRAFT,
)


def _reachable_triangle_task() -> RoleTaskDefinition:
    """Move the initial frame inward so every handle is in its arm workspace."""

    objects = tuple(
        replace(spec, start_position=(-0.10, 0.00, 0.84))
        if spec.name == "triangle_frame"
        else spec
        for spec in _T1_DRAFT.objects
    )
    initial_targets = (
        (-0.25, 0.19, 1.02),
        (0.05, 0.19, 1.02),
        (-0.10, -0.24, 1.02),
    )
    grasp_targets = tuple((x, y, 0.88) for x, y, _ in initial_targets)
    lift_targets = tuple((x, y, 1.12) for x, y, _ in initial_targets)
    phases = (
        replace(_T1_DRAFT.phases[0], arm_targets=initial_targets),
        replace(_T1_DRAFT.phases[1], arm_targets=grasp_targets),
        replace(
            _T1_DRAFT.phases[2],
            arm_targets=lift_targets,
            object_targets=(ObjectKeyframe("triangle_frame", (-0.10, 0.00, 1.08)),),
        ),
        *_T1_DRAFT.phases[3:],
    )
    return replace(_T1_DRAFT, objects=objects, phases=phases)


T1 = _reachable_triangle_task()

_ROLE_TASKS: Mapping[str, RoleTaskDefinition] = {
    task.task_id: task for task in (F1, F2, F3, F4, T1, F5)
}
ROLE_TASK_IDS = tuple(_ROLE_TASKS)


def get_role_task(task_id: str) -> RoleTaskDefinition:
    try:
        return _ROLE_TASKS[task_id]
    except KeyError as error:
        raise ValueError(
            f"unknown task {task_id!r}; choose one of {', '.join(ROLE_TASK_IDS)}"
        ) from error


__all__ = (
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "T1",
    "FOUR_ARM_BASE_XY",
    "GLOBAL_CAMERAS",
    "TABLE_HEIGHT",
    "THREE_ARM_BASE_XY",
    "ObjectKeyframe",
    "PreviewObjectSpec",
    "ROLE_TASK_IDS",
    "RolePhase",
    "RoleTaskDefinition",
    "get_role_task",
)
