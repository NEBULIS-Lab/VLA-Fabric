"""Four-Panda real-contact arch assembly with a physical beam handoff."""

from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CompositeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from multiarm_sim.role_tasks import FOUR_ARM_BASE_XY, get_role_task


GLOBAL_CAMERAS = ("agentview", "frontview")
LOCAL_CAMERAS = tuple(f"robot{i}_eye_in_hand" for i in range(4))
CONTROL_FREQUENCY = 20
TABLE_FULL_SIZE = (1.60, 1.60, 0.05)
TABLE_HEIGHT = 0.80

PILLAR_HALF_WIDTH = 0.025
PILLAR_HALF_HEIGHT = 0.050
PILLAR_GRASP_WIDTH = 2.0 * PILLAR_HALF_WIDTH
BEAM_HALF_LENGTH = 0.150
BEAM_HALF_WIDTH = 0.021
BEAM_HALF_HEIGHT = 0.022
BEAM_GRASP_WIDTH = 2.0 * BEAM_HALF_WIDTH
RECEIVER_GRASP_X_RANGE = (-0.130, -0.100)

PILLAR_TARGET_XY = ((-0.10, -0.16), (0.10, -0.16))
BEAM_HANDOFF_XY = (0.08, -0.31)
INITIAL_XY_BOUNDS = np.array(
    (
        ((-0.42, 0.06), (-0.28, 0.18)),
        ((0.28, 0.06), (0.42, 0.18)),
        ((0.14, -0.46), (0.28, -0.30)),
    ),
    dtype=np.float64,
)
INITIAL_YAW_JITTER = math.radians(4.0)
ASSEMBLY_TARGET_XY_JITTER = 0.010

CONTACT_CONFIRM_STEPS = 3
POSITION_TOLERANCE = 0.025
BEAM_POSITION_TOLERANCE = 0.030
# A slightly wider event window records that both grippers delivered the beam
# to the pillar tops before support contact unloads a fingertip. Final stable
# success still uses BEAM_POSITION_TOLERANCE above.
BEAM_SEATING_CONTACT_TOLERANCE = 0.045
RETREAT_DISTANCE = 0.06
STABLE_LINEAR_SPEED = 0.06
STABLE_ANGULAR_SPEED = 0.30
STABLE_STEPS = 6

ARCH_SUCCESS_EVIDENCE = (
    "initial_grasps_confirmed",
    "pillars_lifted",
    "pillars_placed",
    "beam_handoff_confirmed",
    "beam_seated",
    "dual_release_after_seating",
    "released",
    "retreated",
    "stable",
)


def arch_success_from_state(state) -> bool:
    return all(bool(state.get(name, False)) for name in ARCH_SUCCESS_EVIDENCE)


def _yaw_quat(yaw: float) -> np.ndarray:
    quat_xyzw = T.axisangle2quat(np.array([0.0, 0.0, float(yaw)]))
    return T.convert_quat(quat_xyzw, to="wxyz")


def _look_at_quat(position, target=(0.0, 0.0, 0.92)) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return T.convert_quat(T.mat2quat(np.column_stack([right, up, -forward])), to="wxyz")


def _pillar(name: str, rgba) -> CompositeObject:
    """A plain rectangular pillar grasped directly on its body."""

    grasp_center = [0.0, 0.0, 0.025]
    return CompositeObject(
        name=name,
        total_size=[PILLAR_HALF_WIDTH, PILLAR_HALF_WIDTH, PILLAR_HALF_HEIGHT],
        geom_types=["box"],
        geom_sizes=[[PILLAR_HALF_WIDTH, PILLAR_HALF_WIDTH, PILLAR_HALF_HEIGHT]],
        geom_locations=[[0.0, 0.0, 0.0]],
        geom_quats=[None],
        geom_names=["body"],
        geom_rgbas=[rgba],
        geom_frictions=[[3.0, 0.05, 0.001]],
        geom_condims=[4],
        density=180.0,
        locations_relative_to_center=True,
        joints="default",
        sites=[
            {
                "name": "grasp_site",
                "pos": array_to_string(grasp_center),
                "size": "0.004",
                "rgba": "0 0 0 0",
                "type": "sphere",
            }
        ],
        duplicate_collision_geoms=True,
    )


def _beam(name: str = "beam") -> CompositeObject:
    supplier_center = [0.040, 0.0, 0.0]
    receiver_center = [-0.115, 0.0, 0.0]
    beam_color = [0.10, 0.30, 0.86, 1.0]
    return CompositeObject(
        name=name,
        total_size=[BEAM_HALF_LENGTH, BEAM_HALF_WIDTH, BEAM_HALF_HEIGHT],
        geom_types=["box"],
        geom_sizes=[[BEAM_HALF_LENGTH, BEAM_HALF_WIDTH, BEAM_HALF_HEIGHT]],
        geom_locations=[[0.0, 0.0, 0.0]],
        geom_quats=[None],
        geom_names=["body"],
        geom_rgbas=[beam_color],
        geom_frictions=[[3.0, 0.05, 0.001]],
        geom_condims=[4],
        density=55.0,
        locations_relative_to_center=True,
        joints="default",
        sites=[
            {
                "name": "supplier_site",
                "pos": array_to_string(supplier_center),
                "size": "0.004",
                "rgba": "0 0 0 0",
                "type": "sphere",
            },
            {
                "name": "receiver_site",
                "pos": array_to_string(receiver_center),
                "size": "0.004",
                "rgba": "0 0 0 0",
                "type": "sphere",
            },
        ],
        duplicate_collision_geoms=True,
    )


class FourArmArchAssembly(ManipulationEnv):
    """Assemble two pillars and hand a beam between two real grippers."""

    preview_only = True
    interaction_model = "real_mujoco_contact_v1"

    def __init__(
        self,
        robots=("Panda",) * 4,
        controller_configs=None,
        base_types="NullMount",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        use_object_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_gpu_device_id=-1,
        control_freq=CONTROL_FREQUENCY,
        horizon=1200,
        ignore_done=True,
        hard_reset=True,
        global_image_size=512,
        wrist_image_size=256,
        seed=None,
        receiver_grasp_x=-0.130,
    ):
        self.task = get_role_task("f2_arch_assembly")
        self.table_full_size = np.asarray(TABLE_FULL_SIZE, dtype=float)
        self.table_offset = np.array([0.0, 0.0, TABLE_HEIGHT])
        self.table_friction = (1.0, 0.005, 0.0001)
        self.use_object_obs = use_object_obs
        pillar_z = TABLE_HEIGHT + PILLAR_HALF_HEIGHT
        self.nominal_pillar_targets = np.array(
            [[*xy, pillar_z] for xy in PILLAR_TARGET_XY], dtype=np.float64
        )
        self.nominal_beam_target = np.array(
            [0.0, -0.16, TABLE_HEIGHT + 2.0 * PILLAR_HALF_HEIGHT + BEAM_HALF_HEIGHT],
            dtype=np.float64,
        )
        self.pillar_targets = self.nominal_pillar_targets.copy()
        self.beam_target = self.nominal_beam_target.copy()
        self.beam_handoff_target = np.array(
            [*BEAM_HANDOFF_XY, 1.00], dtype=np.float64
        )
        self.scene_offset = np.zeros(2, dtype=np.float64)
        self.scene_yaw = 0.0
        self.assembly_target_offset = np.zeros(2, dtype=np.float64)
        self.sampled_object_poses = OrderedDict()
        if not RECEIVER_GRASP_X_RANGE[0] <= receiver_grasp_x <= RECEIVER_GRASP_X_RANGE[1]:
            raise ValueError(
                f"receiver_grasp_x must be in {RECEIVER_GRASP_X_RANGE}"
            )
        self.receiver_grasp_x = float(receiver_grasp_x)
        self._reset_task_history()

        cameras = (*GLOBAL_CAMERAS, *LOCAL_CAMERAS)
        camera_sizes = [global_image_size] * 2 + [wrist_image_size] * 4
        super().__init__(
            robots=robots,
            env_configuration="default",
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=cameras,
            camera_heights=camera_sizes,
            camera_widths=camera_sizes,
            camera_depths=[False] * len(cameras),
            camera_segmentations=[None] * len(cameras),
            seed=seed,
        )

    @property
    def instruction(self) -> str:
        return self.task.instruction

    def _load_model(self):
        super()._load_model()
        for robot, (x, y) in zip(self.robots, FOUR_ARM_BASE_XY):
            robot.robot_model.set_base_xpos([x, y, TABLE_HEIGHT])
            robot.robot_model.set_base_ori([0.0, 0.0, math.atan2(-y, -x)])

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0.0, 0.0, 0.0])
        arena.set_camera("agentview", pos=[0.0, 0.0, 2.70], quat=[1.0, 0.0, 0.0, 0.0])
        arena.worldbody.find(".//camera[@name='agentview']").set("fovy", "48")
        front_position = np.array([1.55, -1.65, 1.62])
        arena.set_camera("frontview", pos=front_position, quat=_look_at_quat(front_position))
        arena.worldbody.find(".//camera[@name='frontview']").set("fovy", "52")

        self.left_pillar = _pillar("left_pillar", [0.88, 0.16, 0.12, 1.0])
        self.right_pillar = _pillar("right_pillar", [0.14, 0.72, 0.22, 1.0])
        self.beam = _beam()

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[
                self.left_pillar,
                self.right_pillar,
                self.beam,
            ],
        )

    def _setup_references(self):
        super()._setup_references()
        self.dynamic_objects = OrderedDict(
            left_pillar=self.left_pillar,
            right_pillar=self.right_pillar,
            beam=self.beam,
        )
        self.object_body_ids = {
            name: self.sim.model.body_name2id(model.root_body)
            for name, model in self.dynamic_objects.items()
        }
        site_names = (
            ("left_body", self.left_pillar, "grasp_site"),
            ("right_body", self.right_pillar, "grasp_site"),
            ("beam_supplier", self.beam, "supplier_site"),
            ("beam_receiver", self.beam, "receiver_site"),
        )
        self.grasp_site_ids = OrderedDict(
            (
                label,
                self.sim.model.site_name2id(f"{model.naming_prefix}{site}"),
            )
            for label, model, site in site_names
        )

        dynamic_geom_sets = {
            name: {self.sim.model.geom_name2id(geom) for geom in model.contact_geoms}
            for name, model in self.dynamic_objects.items()
        }
        robot_geoms = {
            self.sim.model.geom_name2id(name)
            for robot in self.robots
            for name in robot.robot_model.contact_geoms
        }
        pairs = set()
        names = tuple(dynamic_geom_sets)
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                pairs.update(
                    frozenset((left, right))
                    for left in dynamic_geom_sets[name]
                    for right in dynamic_geom_sets[other]
                )
            pairs.update(
                frozenset((geom, other))
                for geom in dynamic_geom_sets[name]
                for other in robot_geoms
            )
        self._forbidden_contact_pairs = pairs

    def _set_free_pose(self, model, position, yaw: float) -> None:
        quat = _yaw_quat(yaw)
        self.sim.data.set_joint_qpos(model.joints[0], np.concatenate([position, quat]))
        joint_id = self.sim.model.joint_name2id(model.joints[0])
        dof_start = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_start : dof_start + 6] = 0.0

    def _reset_internal(self):
        super()._reset_internal()
        initial_xy = self.rng.uniform(
            INITIAL_XY_BOUNDS[:, 0, :],
            INITIAL_XY_BOUNDS[:, 1, :],
        )
        yaws = self.rng.uniform(-INITIAL_YAW_JITTER, INITIAL_YAW_JITTER, size=3)
        self.assembly_target_offset = self.rng.uniform(
            -ASSEMBLY_TARGET_XY_JITTER,
            ASSEMBLY_TARGET_XY_JITTER,
            size=2,
        )
        self.pillar_targets = self.nominal_pillar_targets.copy()
        self.pillar_targets[:, :2] += self.assembly_target_offset
        self.beam_target = self.nominal_beam_target.copy()
        self.beam_target[:2] += self.assembly_target_offset
        receiver_site_id = self.grasp_site_ids["beam_receiver"]
        self.sim.model.site_pos[receiver_site_id] = np.array(
            [self.receiver_grasp_x, 0.0, 0.0]
        )
        pillar_z = TABLE_HEIGHT + PILLAR_HALF_HEIGHT + 0.003
        for model, xy, yaw in zip(
            (self.left_pillar, self.right_pillar),
            initial_xy[:2],
            yaws[:2],
        ):
            self._set_free_pose(model, np.array([*xy, pillar_z]), yaw)
        self._set_free_pose(
            self.beam,
            np.array([*initial_xy[2], TABLE_HEIGHT + BEAM_HALF_HEIGHT + 0.003]),
            yaws[2],
        )
        region_centers = INITIAL_XY_BOUNDS.mean(axis=1)
        self.scene_offset = (initial_xy - region_centers).mean(axis=0)
        self.scene_yaw = float(yaws.mean())
        self._reset_task_history()
        self.sim.forward()
        self.sampled_object_poses = OrderedDict(
            (name, pose.copy()) for name, pose in self.object_poses().items()
        )
        self._initial_forbidden_contacts = self._count_forbidden_contacts()

    def object_poses(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict(
            (
                name,
                np.concatenate(
                    [
                        np.asarray(self.sim.data.body_xpos[body_id]),
                        np.asarray(self.sim.data.body_xquat[body_id]),
                    ]
                ).copy(),
            )
            for name, body_id in self.object_body_ids.items()
        )

    def grasp_sites(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict(
            (name, np.asarray(self.sim.data.site_xpos[site_id]).copy())
            for name, site_id in self.grasp_site_ids.items()
        )

    def assigned_grasps(self) -> tuple[bool, bool, bool, bool]:
        geom_names = (
            f"{self.left_pillar.naming_prefix}body",
            f"{self.right_pillar.naming_prefix}body",
            f"{self.beam.naming_prefix}body",
            f"{self.beam.naming_prefix}body",
        )
        return tuple(
            bool(self._check_grasp(robot.gripper, geom_name))
            for robot, geom_name in zip(self.robots, geom_names)
        )

    def _count_forbidden_contacts(self) -> int:
        return sum(
            frozenset((int(contact.geom1), int(contact.geom2))) in self._forbidden_contact_pairs
            for contact in self.sim.data.contact[: self.sim.data.ncon]
        )

    def initial_forbidden_contacts(self) -> int:
        return int(getattr(self, "_initial_forbidden_contacts", self._count_forbidden_contacts()))

    def _reset_task_history(self) -> None:
        self._history = {name: False for name in ARCH_SUCCESS_EVIDENCE}
        self._initial_grasp_steps = 0
        self._handoff_steps = 0
        self._stable_steps = 0
        self._success_latched = False
        self._last_task_update_time = float("-inf")
        self._latest_task_state = {}

    def _task_state(self) -> dict[str, object]:
        now = float(self.sim.data.time)
        if now == self._last_task_update_time and self._latest_task_state:
            return dict(self._latest_task_state)

        poses = self.object_poses()
        grasps = self.assigned_grasps()
        initial_now = bool(grasps[0] and grasps[1] and grasps[2])
        self._initial_grasp_steps = self._initial_grasp_steps + 1 if initial_now else 0
        if self._initial_grasp_steps >= CONTACT_CONFIRM_STEPS:
            self._history["initial_grasps_confirmed"] = True

        pillar_bottoms = np.array(
            [poses["left_pillar"][2], poses["right_pillar"][2]]
        ) - PILLAR_HALF_HEIGHT
        if (
            self._history["initial_grasps_confirmed"]
            and np.min(pillar_bottoms) >= TABLE_HEIGHT + 0.05
        ):
            self._history["pillars_lifted"] = True
        pillar_errors = np.linalg.norm(
            np.stack([poses["left_pillar"][:3], poses["right_pillar"][:3]])
            - self.pillar_targets,
            axis=1,
        )
        if self._history["pillars_lifted"] and np.max(pillar_errors) <= POSITION_TOLERANCE:
            self._history["pillars_placed"] = True

        handoff_now = bool(grasps[2] and grasps[3])
        self._handoff_steps = self._handoff_steps + 1 if handoff_now else 0
        if self._history["pillars_placed"] and self._handoff_steps >= CONTACT_CONFIRM_STEPS:
            self._history["beam_handoff_confirmed"] = True
        beam_position = poses["beam"][:3]
        beam_error = float(np.linalg.norm(beam_position - self.beam_target))
        pillars_within_tolerance = bool(np.max(pillar_errors) <= POSITION_TOLERANCE)
        if (
            self._history["beam_handoff_confirmed"]
            and grasps[2]
            and grasps[3]
            and pillars_within_tolerance
            and beam_error <= BEAM_SEATING_CONTACT_TOLERANCE
        ):
            self._history["beam_seated"] = True
        if self._history["beam_seated"] and not grasps[2] and not grasps[3]:
            self._history["dual_release_after_seating"] = True
        if self._history["dual_release_after_seating"] and not any(grasps):
            self._history["released"] = True

        retreat_distances = tuple(
            float(np.linalg.norm(
                np.asarray(self.sim.data.site_xpos[robot.eef_site_id["right"]]) - site
            ))
            for robot, site in zip(self.robots, self.grasp_sites().values())
        )
        if self._history["released"] and min(retreat_distances) >= RETREAT_DISTANCE:
            self._history["retreated"] = True

        linear_speeds = tuple(
            float(np.linalg.norm(self.sim.data.get_body_xvelp(model.root_body)))
            for model in self.dynamic_objects.values()
        )
        angular_speeds = tuple(
            float(np.linalg.norm(self.sim.data.get_body_xvelr(model.root_body)))
            for model in self.dynamic_objects.values()
        )
        stable_now = bool(
            self._history["retreated"]
            and pillars_within_tolerance
            and beam_error <= BEAM_POSITION_TOLERANCE
            and max(linear_speeds) <= STABLE_LINEAR_SPEED
            and max(angular_speeds) <= STABLE_ANGULAR_SPEED
        )
        self._stable_steps = self._stable_steps + 1 if stable_now else 0
        if self._stable_steps >= STABLE_STEPS:
            self._history["stable"] = True
        self._success_latched |= arch_success_from_state(self._history)

        state = {
            "task_id": self.task.task_id,
            "preview_only": True,
            "interaction_model": self.interaction_model,
            "scene_offset": self.scene_offset.copy(),
            "scene_yaw": self.scene_yaw,
            "assembly_target_offset": self.assembly_target_offset.copy(),
            "sampled_object_poses": OrderedDict(
                (name, pose.copy())
                for name, pose in self.sampled_object_poses.items()
            ),
            "receiver_grasp_x": self.receiver_grasp_x,
            "initial_forbidden_contacts": self.initial_forbidden_contacts(),
            "assigned_grasps": grasps,
            "pillar_errors": pillar_errors.copy(),
            "beam_error": beam_error,
            "beam_position": beam_position.copy(),
            "retreat_distances": retreat_distances,
            "linear_speeds": linear_speeds,
            "angular_speeds": angular_speeds,
            **self._history,
            "success": self._success_latched,
        }
        self._last_task_update_time = now
        self._latest_task_state = state
        return dict(state)

    def task_status(self) -> dict[str, object]:
        return self._task_state()

    def _check_success(self):
        return bool(self._task_state()["success"])

    def reward(self, action=None):
        state = self._task_state()
        return 1.0 if state["success"] else 0.04 * sum(state["assigned_grasps"])


def make_arch_assembly_env(
    *,
    global_image_size: int = 512,
    wrist_image_size: int = 256,
    horizon: int = 1200,
    seed: int | None = None,
    receiver_grasp_x: float = -0.130,
):
    return FourArmArchAssembly(
        global_image_size=global_image_size,
        wrist_image_size=wrist_image_size,
        horizon=horizon,
        seed=seed,
        receiver_grasp_x=receiver_grasp_x,
    )


__all__ = (
    "BEAM_GRASP_WIDTH",
    "BEAM_HALF_LENGTH",
    "BEAM_HALF_WIDTH",
    "GLOBAL_CAMERAS",
    "LOCAL_CAMERAS",
    "PILLAR_GRASP_WIDTH",
    "PILLAR_HALF_WIDTH",
    "FourArmArchAssembly",
    "arch_success_from_state",
    "make_arch_assembly_env",
)
