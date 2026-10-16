"""Declarative contracts for short role-diverse multi-arm task previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


TABLE_HEIGHT = 0.80
FOUR_ARM_BASE_XY = (
    (-0.56, 0.56),
    (0.56, 0.56),
    (0.56, -0.56),
    (-0.56, -0.56),
)
THREE_ARM_BASE_XY = (
    (-0.52, 0.45),
    (0.52, 0.45),
    (0.00, 0.45 - 1.04 * 3**0.5 / 2),
)
GLOBAL_CAMERAS = ("agentview", "frontview")


@dataclass(frozen=True)
class PreviewObjectSpec:
    name: str
    shape: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    start_position: tuple[float, float, float]
    start_yaw: float = 0.0
    movable: bool = True


@dataclass(frozen=True)
class ObjectKeyframe:
    name: str
    position: tuple[float, float, float]
    yaw: float = 0.0


@dataclass(frozen=True)
class RolePhase:
    name: str
    steps: int
    arm_targets: tuple[tuple[float, float, float], ...]
    grippers_closed: tuple[bool, ...]
    object_targets: tuple[ObjectKeyframe, ...] = ()
    axis_angle_commands: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class RoleTaskDefinition:
    task_id: str
    title: str
    instruction: str
    num_agents: int
    base_xy: tuple[tuple[float, float], ...]
    role_instructions: tuple[str, ...]
    handoff_count: int
    expected_steps: int
    max_steps: int
    objects: tuple[PreviewObjectSpec, ...]
    phases: tuple[RolePhase, ...]
    global_cameras: tuple[str, ...] = GLOBAL_CAMERAS

    def __post_init__(self) -> None:
        if len(self.base_xy) != self.num_agents:
            raise ValueError("base_xy must contain one base per agent")
        if len(self.role_instructions) != self.num_agents:
            raise ValueError("role_instructions must contain one role per agent")
        if sum(phase.steps for phase in self.phases) != self.expected_steps:
            raise ValueError("phase steps must sum to expected_steps")
        object_names = {spec.name for spec in self.objects}
        if len(object_names) != len(self.objects):
            raise ValueError("object names must be unique")
        for phase in self.phases:
            if phase.steps <= 0:
                raise ValueError("phase steps must be positive")
            if len(phase.arm_targets) != self.num_agents:
                raise ValueError(f"{phase.name} must target every agent")
            if len(phase.grippers_closed) != self.num_agents:
                raise ValueError(f"{phase.name} must command every gripper")
            if phase.axis_angle_commands and len(phase.axis_angle_commands) != self.num_agents:
                raise ValueError(f"{phase.name} rotations must target every agent")
            unknown = {target.name for target in phase.object_targets} - object_names
            if unknown:
                raise ValueError(f"{phase.name} references unknown objects: {sorted(unknown)}")


def _obj(
    name: str,
    shape: str,
    size: tuple[float, ...],
    rgba: tuple[float, float, float, float],
    position: tuple[float, float, float],
    *,
    movable: bool = True,
) -> PreviewObjectSpec:
    return PreviewObjectSpec(name, shape, size, rgba, position, movable=movable)


def _key(name: str, xyz: tuple[float, float, float], yaw: float = 0.0) -> ObjectKeyframe:
    return ObjectKeyframe(name, xyz, yaw)


F1 = RoleTaskDefinition(
    task_id="f1_drawer_handover",
    title="F1 抽屉取物—单次交接—入盒",
    instruction=(
        "Open the drawer and receiving cover in parallel, retrieve the yellow block, "
        "handoff exactly once, deposit it, then close both mechanisms."
    ),
    num_agents=4,
    base_xy=FOUR_ARM_BASE_XY,
    role_instructions=(
        "Open, hold, and close the orange drawer.",
        "Retrieve and present the yellow block.",
        "Receive the block once and deposit it in the blue box.",
        "Open, hold, and close the blue box cover.",
    ),
    handoff_count=1,
    expected_steps=255,
    max_steps=500,
    objects=(
        _obj("cabinet", "box", (0.20, 0.16, 0.04), (0.35, 0.37, 0.42, 1), (-0.24, 0.24, 0.84), movable=False),
        _obj("drawer", "box", (0.16, 0.13, 0.025), (0.94, 0.42, 0.08, 1), (-0.24, 0.23, 0.895)),
        _obj("block", "box", (0.035, 0.035, 0.035), (0.95, 0.82, 0.10, 1), (-0.24, 0.19, 0.955)),
        _obj("box", "box", (0.17, 0.14, 0.045), (0.10, 0.36, 0.90, 1), (0.07, -0.24, 0.85), movable=False),
        _obj("cover", "box", (0.17, 0.14, 0.015), (0.16, 0.62, 0.94, 1), (0.07, -0.24, 0.925)),
    ),
    phases=(
        RolePhase(
            "open", 50,
            ((-0.24, 0.35, 0.93), (-0.24, 0.19, 1.04), (0.02, 0.00, 1.02), (-0.10, -0.24, 0.97)),
            (True, False, False, True),
            (_key("drawer", (-0.24, 0.34, 0.895)), _key("cover", (-0.20, -0.24, 0.925))),
        ),
        RolePhase(
            "retrieve", 50,
            ((-0.24, 0.35, 0.93), (-0.13, 0.10, 1.03), (0.03, 0.00, 1.02), (-0.20, -0.24, 0.97)),
            (True, True, False, True),
            (_key("block", (-0.13, 0.10, 1.00)),),
        ),
        RolePhase(
            "handoff", 55,
            ((-0.24, 0.35, 0.93), (-0.025, 0.025, 1.02), (0.035, -0.025, 1.02), (-0.20, -0.24, 0.97)),
            (True, True, True, True),
            (_key("block", (0.005, 0.00, 1.00)),),
        ),
        RolePhase(
            "deposit", 55,
            ((-0.24, 0.23, 0.93), (-0.36, 0.35, 1.10), (0.07, -0.24, 0.96), (-0.20, -0.24, 0.97)),
            (True, False, True, True),
            (_key("drawer", (-0.24, 0.23, 0.895)), _key("block", (0.07, -0.24, 0.915))),
        ),
        RolePhase(
            "close", 45,
            ((-0.45, 0.45, 1.12), (-0.40, 0.36, 1.12), (0.40, -0.40, 1.12), (0.07, -0.24, 0.97)),
            (False, False, False, True),
            (_key("cover", (0.07, -0.24, 0.925)),),
        ),
    ),
)


F2 = RoleTaskDefinition(
    task_id="f2_arch_assembly",
    title="F2 双线并行拱门组装",
    instruction=(
        "Place two plain pillar blocks, return the upper arms to neutral, "
        "then have both lower arms jointly lower and release the plain beam."
    ),
    num_agents=4,
    base_xy=FOUR_ARM_BASE_XY,
    role_instructions=(
        "Directly grasp and place the red left pillar, then return to neutral.",
        "Directly grasp and place the green right pillar, then return to neutral.",
        "Directly grasp and supply the blue beam, then keep holding it for joint seating and release.",
        "Directly receive the blue beam, jointly seat it on both pillars, and release with the supplier.",
    ),
    handoff_count=1,
    expected_steps=250,
    max_steps=500,
    objects=(
        _obj("left_pillar", "box", (0.05, 0.05, 0.10), (0.88, 0.16, 0.12, 1), (-0.35, -0.02, 0.85)),
        _obj("right_pillar", "box", (0.05, 0.05, 0.10), (0.14, 0.72, 0.22, 1), (0.35, -0.02, 0.85)),
        _obj("beam", "box", (0.30, 0.042, 0.044), (0.10, 0.30, 0.86, 1), (0.20, -0.40, 0.822)),
    ),
    phases=(
        RolePhase("pick", 50, ((-0.34, 0.20, 1.06), (0.34, 0.20, 1.06), (0.27, -0.30, 0.95), (0.03, -0.12, 1.10)), (True, True, True, False)),
        RolePhase(
            "parallel_place", 55,
            ((-0.18, -0.05, 1.10), (0.18, -0.05, 1.10), (0.08, -0.10, 1.12), (-0.01, -0.08, 1.12)),
            (True, True, True, False),
            (_key("left_pillar", (-0.18, -0.05, 0.965)), _key("right_pillar", (0.18, -0.05, 0.965)), _key("beam", (0.08, -0.10, 1.08))),
        ),
        RolePhase(
            "handoff", 45,
            ((-0.18, -0.05, 1.08), (0.18, -0.05, 1.08), (0.03, -0.08, 1.12), (-0.03, -0.08, 1.12)),
            (True, True, True, True),
            (_key("beam", (0.00, -0.08, 1.08)),),
        ),
        RolePhase(
            "seat", 55,
            ((-0.18, -0.05, 1.05), (0.18, -0.05, 1.05), (0.00, -0.05, 1.16), (0.00, -0.05, 1.13)),
            (False, False, True, True),
            (_key("beam", (0.00, -0.05, 1.145)),),
        ),
        RolePhase("retreat", 45, ((-0.42, 0.40, 1.15), (0.42, 0.40, 1.15), (0.42, -0.40, 1.15), (-0.42, -0.40, 1.15)), (False, False, False, False)),
    ),
)


F3 = RoleTaskDefinition(
    task_id="f3_load_close",
    title="F3 双物体装料—推盒—合盖",
    instruction="Load both colored parts, insert the tray, and close the cover.",
    num_agents=4,
    base_xy=FOUR_ARM_BASE_XY,
    role_instructions=(
        "Load the red part into the red slot.",
        "Load the blue part into the blue slot.",
        "Pull the tray for loading and push it into the housing.",
        "Open and close the yellow cover.",
    ),
    handoff_count=0,
    expected_steps=240,
    max_steps=500,
    objects=(
        _obj("red_part", "box", (0.04, 0.04, 0.04), (0.90, 0.12, 0.10, 1), (-0.30, 0.20, 0.86)),
        _obj("blue_part", "box", (0.04, 0.04, 0.04), (0.10, 0.32, 0.92, 1), (0.30, 0.20, 0.86)),
        _obj("tray", "box", (0.25, 0.16, 0.025), (0.55, 0.57, 0.62, 1), (0.00, -0.02, 0.84)),
        _obj("housing", "box", (0.28, 0.19, 0.045), (0.20, 0.23, 0.28, 1), (0.00, -0.28, 0.85), movable=False),
        _obj("cover", "box", (0.27, 0.18, 0.018), (0.95, 0.72, 0.10, 1), (0.00, -0.28, 0.93)),
    ),
    phases=(
        RolePhase(
            "open_and_pick", 60,
            ((-0.30, 0.20, 0.94), (0.30, 0.20, 0.94), (0.00, 0.12, 0.91), (-0.23, -0.28, 0.97)),
            (True, True, True, True),
            (_key("tray", (0.00, 0.12, 0.84)), _key("cover", (-0.28, -0.28, 0.93))),
        ),
        RolePhase(
            "load", 65,
            ((-0.09, 0.12, 0.93), (0.09, 0.12, 0.93), (0.00, 0.23, 0.91), (-0.28, -0.28, 0.97)),
            (True, True, True, True),
            (_key("red_part", (-0.09, 0.12, 0.885)), _key("blue_part", (0.09, 0.12, 0.885))),
        ),
        RolePhase(
            "insert_tray", 60,
            ((-0.35, 0.34, 1.12), (0.35, 0.34, 1.12), (0.00, -0.28, 0.91), (-0.28, -0.28, 0.97)),
            (False, False, True, True),
            (_key("tray", (0.00, -0.28, 0.84)), _key("red_part", (-0.09, -0.28, 0.885)), _key("blue_part", (0.09, -0.28, 0.885))),
        ),
        RolePhase(
            "close_cover", 55,
            ((-0.42, 0.42, 1.15), (0.42, 0.42, 1.15), (0.42, -0.42, 1.15), (0.00, -0.28, 0.97)),
            (False, False, False, True),
            (_key("cover", (0.00, -0.28, 0.93)),),
        ),
    ),
)


F4 = RoleTaskDefinition(
    task_id="f4_shaft_lock",
    title="F4 导向轴插接与旋转锁紧",
    instruction="Stabilize both guides, insert the keyed shaft, and rotate the lock cap.",
    num_agents=4,
    base_xy=FOUR_ARM_BASE_XY,
    role_instructions=(
        "Stabilize the red west guide handle.",
        "Align and insert the blue keyed shaft.",
        "Stabilize the green east guide handle.",
        "Rotate the yellow locking cap after insertion.",
    ),
    handoff_count=0,
    expected_steps=250,
    max_steps=500,
    objects=(
        _obj("socket", "cylinder", (0.09, 0.07), (0.24, 0.27, 0.34, 1), (0.00, 0.00, 0.87), movable=False),
        _obj("guide", "box", (0.24, 0.07, 0.025), (0.60, 0.62, 0.68, 1), (0.00, 0.00, 0.99), movable=False),
        _obj("west_handle", "box", (0.07, 0.035, 0.035), (0.90, 0.14, 0.10, 1), (-0.29, 0.00, 0.99), movable=False),
        _obj("east_handle", "box", (0.07, 0.035, 0.035), (0.12, 0.72, 0.20, 1), (0.29, 0.00, 0.99), movable=False),
        _obj("shaft", "cylinder", (0.045, 0.18), (0.12, 0.34, 0.92, 1), (0.00, 0.28, 1.02)),
        _obj("lock_cap", "cylinder", (0.075, 0.025), (0.96, 0.73, 0.10, 1), (0.00, -0.20, 0.87)),
    ),
    phases=(
        RolePhase("grasp_roles", 50, ((-0.29, 0.00, 1.03), (0.00, 0.28, 1.17), (0.29, 0.00, 1.03), (0.00, -0.20, 0.94)), (True, True, True, True)),
        RolePhase(
            "align", 55,
            ((-0.27, 0.00, 1.03), (0.00, 0.00, 1.22), (0.27, 0.00, 1.03), (0.00, -0.08, 1.02)),
            (True, True, True, True),
            (_key("shaft", (0.00, 0.00, 1.08)), _key("lock_cap", (0.00, -0.08, 0.98))),
        ),
        RolePhase(
            "insert", 55,
            ((-0.27, 0.00, 1.03), (0.00, 0.00, 1.04), (0.27, 0.00, 1.03), (0.00, -0.04, 1.06)),
            (True, True, True, True),
            (_key("shaft", (0.00, 0.00, 0.98)), _key("lock_cap", (0.00, 0.00, 1.085))),
        ),
        RolePhase(
            "rotate_lock", 50,
            ((-0.27, 0.00, 1.03), (0.00, 0.12, 1.17), (0.27, 0.00, 1.03), (0.00, 0.00, 1.10)),
            (True, False, True, True),
            (_key("lock_cap", (0.00, 0.00, 1.085), 1.0472),),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.35)),
        ),
        RolePhase("retreat", 40, ((-0.42, 0.42, 1.16), (0.40, 0.42, 1.16), (0.42, -0.42, 1.16), (-0.42, -0.42, 1.16)), (False, False, False, False)),
    ),
)


T1 = RoleTaskDefinition(
    task_id="t1_triangle_frame",
    title="T1 三臂三角框套柱",
    instruction="Use three arms to lift the triangular frame and lower it around the post.",
    num_agents=3,
    base_xy=THREE_ARM_BASE_XY,
    role_instructions=(
        "Grasp the northwest red triangle handle.",
        "Grasp the northeast green triangle handle.",
        "Grasp the south blue triangle handle.",
    ),
    handoff_count=0,
    expected_steps=270,
    max_steps=500,
    objects=(
        _obj("triangle_frame", "triangle_frame", (0.42, 0.25, 0.045), (0.95, 0.45, 0.08, 1), (-0.27, 0.00, 0.84)),
        _obj("triangle_post", "triangle_post", (0.17, 0.16), (0.14, 0.36, 0.88, 1), (0.08, 0.00, 0.88), movable=False),
    ),
    phases=(
        RolePhase("approach", 45, ((-0.42, 0.19, 1.02), (-0.12, 0.19, 1.02), (-0.27, -0.24, 1.02)), (False, False, False)),
        RolePhase("grasp", 45, ((-0.42, 0.19, 0.88), (-0.12, 0.19, 0.88), (-0.27, -0.24, 0.88)), (True, True, True)),
        RolePhase(
            "lift", 45,
            ((-0.42, 0.19, 1.12), (-0.12, 0.19, 1.12), (-0.27, -0.24, 1.12)),
            (True, True, True),
            (_key("triangle_frame", (-0.27, 0.00, 1.08)),),
        ),
        RolePhase(
            "transport", 50,
            ((-0.07, 0.19, 1.12), (0.23, 0.19, 1.12), (0.08, -0.24, 1.12)),
            (True, True, True),
            (_key("triangle_frame", (0.08, 0.00, 1.08)),),
        ),
        RolePhase(
            "lower", 45,
            ((-0.07, 0.19, 0.91), (0.23, 0.19, 0.91), (0.08, -0.24, 0.91)),
            (True, True, True),
            (_key("triangle_frame", (0.08, 0.00, 0.845)),),
        ),
        RolePhase("release", 40, ((-0.38, 0.38, 1.15), (0.38, 0.38, 1.15), (0.00, -0.43, 1.15)), (False, False, False)),
    ),
)


F5 = RoleTaskDefinition(
    task_id="f5_cooperative_loading",
    title="F5 四臂协同悬持装载",
    instruction=(
        "Use the upper pair to lift and continuously suspend the handled "
        "container while the lower pair loads and releases the blue payload."
    ),
    num_agents=4,
    base_xy=FOUR_ARM_BASE_XY,
    role_instructions=(
        "Grasp and continuously suspend the container's left carry grip.",
        "Grasp and continuously suspend the container's right carry grip.",
        "Grasp the payload's right region, lower it, release it, and retreat.",
        "Grasp the payload's left region, lower it, release it, and retreat.",
    ),
    handoff_count=0,
    expected_steps=280,
    max_steps=400,
    objects=(
        _obj(
            "container",
            "handled_container",
            (0.34, 0.24, 0.07),
            (0.95, 0.68, 0.12, 1.0),
            (0.00, 0.29, 0.85),
        ),
        _obj(
            "payload",
            "box",
            (0.23, 0.04, 0.04),
            (0.10, 0.34, 0.90, 1.0),
            (0.00, -0.31, 0.82),
        ),
    ),
    phases=(
        RolePhase("approach", 24, ((-0.20, 0.29, 1.05), (0.20, 0.29, 1.05), (0.075, -0.31, 1.02), (-0.075, -0.31, 1.02)), (False, False, False, False)),
        RolePhase("descend", 24, ((-0.20, 0.29, 0.88), (0.20, 0.29, 0.88), (0.075, -0.31, 0.86), (-0.075, -0.31, 0.86)), (False, False, False, False)),
        RolePhase("close_confirm", 24, ((-0.20, 0.29, 0.88), (0.20, 0.29, 0.88), (0.075, -0.31, 0.86), (-0.075, -0.31, 0.86)), (True, True, True, True)),
        RolePhase("dual_lift", 36, ((-0.20, 0.29, 1.05), (0.20, 0.29, 1.05), (0.075, -0.31, 1.03), (-0.075, -0.31, 1.03)), (True, True, True, True)),
        RolePhase("rendezvous", 44, ((-0.20, 0.02, 1.05), (0.20, 0.02, 1.05), (0.075, -0.04, 1.12), (-0.075, -0.04, 1.12)), (True, True, True, True)),
        RolePhase("relative_align", 28, ((-0.20, 0.02, 1.05), (0.20, 0.02, 1.05), (0.075, 0.02, 1.10), (-0.075, 0.02, 1.10)), (True, True, True, True)),
        RolePhase("lower_payload", 44, ((-0.20, 0.02, 1.05), (0.20, 0.02, 1.05), (0.075, 0.02, 0.99), (-0.075, 0.02, 0.99)), (True, True, True, True)),
        RolePhase("release_and_retreat", 28, ((-0.20, 0.02, 1.05), (0.20, 0.02, 1.05), (0.24, -0.22, 1.16), (-0.24, -0.22, 1.16)), (True, True, False, False)),
        RolePhase("stabilize", 28, ((-0.20, 0.02, 1.05), (0.20, 0.02, 1.05), (0.36, -0.34, 1.18), (-0.36, -0.34, 1.18)), (True, True, False, False)),
    ),
)


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
