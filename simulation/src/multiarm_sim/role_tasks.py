"""Public role-task contracts with final reachable targets and base layouts."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from multiarm_sim._role_tasks_reachable import (
    F1, F2, F3 as _F3_DRAFT, F4 as _F4_DRAFT, F5,
    FOUR_ARM_BASE_XY, GLOBAL_CAMERAS, TABLE_HEIGHT,
    ObjectKeyframe, PreviewObjectSpec, RolePhase, RoleTaskDefinition,
    T1 as _T1_DRAFT,
)


THREE_ARM_BASE_XY = (
    (-0.62, 0.45),
    (0.62, 0.45),
    (0.00, 0.45 - 1.24 * 3**0.5 / 2),
)


def _reachable_load_task() -> RoleTaskDefinition:
    phases = list(_F3_DRAFT.phases)
    targets = list(phases[1].arm_targets)
    targets[2] = (0.00, 0.12, 0.91)
    phases[1] = replace(phases[1], arm_targets=tuple(targets))
    targets = list(phases[2].arm_targets)
    targets[2] = (0.00, -0.20, 0.95)
    phases[2] = replace(phases[2], arm_targets=tuple(targets))
    return replace(_F3_DRAFT, phases=tuple(phases))


def _reachable_shaft_task() -> RoleTaskDefinition:
    phases = list(_F4_DRAFT.phases)
    targets = list(phases[1].arm_targets)
    # Close above the shaft before descending, avoiding the ambiguous unloaded
    # open aperture observed when closing directly beside the shaft top.
    targets[1] = (0.00, 0.00, 1.31)
    phases[1] = replace(phases[1], arm_targets=tuple(targets))
    targets = list(phases[2].arm_targets)
    targets[1] = (0.00, 0.00, 1.17)
    targets[3] = (0.05, -0.12, 1.20)
    phases[2] = replace(phases[2], arm_targets=tuple(targets))
    targets = list(phases[3].arm_targets)
    targets[3] = (0.02, -0.05, 1.20)
    phases[3] = replace(phases[3], arm_targets=tuple(targets))
    return replace(_F4_DRAFT, phases=tuple(phases))


F3 = _reachable_load_task()
F4 = _reachable_shaft_task()
T1 = replace(_T1_DRAFT, base_xy=THREE_ARM_BASE_XY)
_ROLE_TASKS: Mapping[str, RoleTaskDefinition] = {
    task.task_id: task for task in (F1, F2, F3, F4, T1, F5)
}
ROLE_TASK_IDS = tuple(_ROLE_TASKS)
# These tasks use phase-keyframe semantic previews. F5 uses its dedicated
# real-contact environment and expert with closed-loop completion timing.
SEMANTIC_PREVIEW_TASK_IDS = tuple(
    task_id for task_id in ROLE_TASK_IDS if task_id != "f5_cooperative_loading"
)


def get_role_task(task_id: str) -> RoleTaskDefinition:
    try:
        return _ROLE_TASKS[task_id]
    except KeyError as error:
        raise ValueError(
            f"unknown task {task_id!r}; choose one of {', '.join(ROLE_TASK_IDS)}"
        ) from error


__all__ = (
    "F1", "F2", "F3", "F4", "F5", "T1", "FOUR_ARM_BASE_XY",
    "GLOBAL_CAMERAS", "TABLE_HEIGHT", "THREE_ARM_BASE_XY",
    "ObjectKeyframe", "PreviewObjectSpec", "ROLE_TASK_IDS", "RolePhase",
    "RoleTaskDefinition", "SEMANTIC_PREVIEW_TASK_IDS", "get_role_task",
)
