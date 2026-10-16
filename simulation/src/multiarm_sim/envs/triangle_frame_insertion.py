"""Three-Panda real-contact triangular frame insertion."""

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

from multiarm_sim.role_tasks import THREE_ARM_BASE_XY, get_role_task


GLOBAL_CAMERAS = ("agentview", "frontview")
LOCAL_CAMERAS = tuple(f"robot{i}_eye_in_hand" for i in range(3))
HANDLE_ROLES = ("northwest", "northeast", "south")
CONTROL_FREQUENCY = 20
TABLE_FULL_SIZE = (1.60, 1.60, 0.05)
TABLE_HEIGHT = 0.80

FRAME_SIDE = 0.38
FRAME_RAIL_THICKNESS = 0.04
FRAME_HEIGHT = 0.045
FRAME_INNER_SIDE = FRAME_SIDE - 2.0 * math.sqrt(3.0) * FRAME_RAIL_THICKNESS
POST_SIDE = 0.13
POST_HEIGHT = 0.18
POST_X = 0.0
FRAME_INITIAL_X_RANGES = {
    "left": (-0.280, -0.245),
    "right": (0.245, 0.280),
}
FRAME_INITIAL_Y_RANGE = (-0.05, 0.05)
FRAME_INITIAL_YAW_RANGE = math.radians(4.0)

GRASP_CONFIRM_STEPS = 3
CENTER_TOLERANCE = 0.025
TILT_TOLERANCE_RADIANS = math.radians(6.0)
PLACEMENT_HEIGHT_TOLERANCE = 0.010
RETREAT_DISTANCE = 0.05
STABLE_LINEAR_SPEED = 0.025
STABLE_ANGULAR_SPEED = 0.12
STABLE_STEPS = 8

TRIANGLE_SUCCESS_EVIDENCE = (
    "three_grasp_confirmed",
    "lifted",
    "transported",
    "placed",
    "released",
    "retreated",
    "stable",
)


def triangle_success_from_state(state) -> bool:
    """Return true only when every ordered real-contact milestone occurred."""

    return all(bool(state.get(name, False)) for name in TRIANGLE_SUCCESS_EVIDENCE)


def _yaw_quat(yaw: float) -> np.ndarray:
    quat_xyzw = T.axisangle2quat(np.array([0.0, 0.0, float(yaw)]))
    return T.convert_quat(quat_xyzw, to="wxyz")


def _look_at_quat(position, target=(0.0, 0.0, 0.88)) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    return T.convert_quat(T.mat2quat(rotation), to="wxyz")


def _triangle_vertices(side: float) -> np.ndarray:
    height = side * math.sqrt(3.0) / 2.0
    return np.array(
        [
            [-side / 2.0, height / 3.0, 0.0],
            [side / 2.0, height / 3.0, 0.0],
            [0.0, -2.0 * height / 3.0, 0.0],
        ],
        dtype=np.float64,
    )


def _triangle_frame(name: str = "triangle_frame") -> CompositeObject:
    vertices = _triangle_vertices(FRAME_SIDE)
    locations = []
    sizes = []
    quats = []
    names = []
    rgbas = []
    for index, (start, end) in enumerate(
        ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0]))
    ):
        delta = end - start
        yaw = math.atan2(float(delta[1]), float(delta[0]))
        locations.append(((start + end) / 2.0).tolist())
        sizes.append([FRAME_SIDE / 2.0, FRAME_RAIL_THICKNESS / 2.0, FRAME_HEIGHT / 2.0])
        quats.append(_yaw_quat(yaw))
        names.append(f"rail_{index}")
        rgbas.append([0.95, 0.45, 0.08, 1.0])

    role_colors = (
        [0.90, 0.14, 0.10, 1.0],
        [0.12, 0.72, 0.20, 1.0],
        [0.12, 0.34, 0.92, 1.0],
    )
    handle_length = 0.12
    handle_width = 0.04
    sites = []
    for role, vertex, color in zip(HANDLE_ROLES, vertices, role_colors):
        outward = vertex[:2] / np.linalg.norm(vertex[:2])
        center = vertex.copy()
        center[:2] += outward * handle_length / 2.0
        yaw = math.atan2(float(outward[1]), float(outward[0]))
        locations.append(center.tolist())
        sizes.append([handle_length / 2.0, handle_width / 2.0, handle_width / 2.0])
        quats.append(_yaw_quat(yaw))
        names.append(f"handle_{role}")
        rgbas.append(color)
        sites.append(
            {
                "name": f"handle_{role}_site",
                "pos": array_to_string(center),
                "size": "0.008",
                "rgba": array_to_string(color),
                "type": "sphere",
            }
        )

    bound = FRAME_SIDE / 2.0 + handle_length + 0.02
    return CompositeObject(
        name=name,
        total_size=[bound, bound, FRAME_HEIGHT / 2.0],
        geom_types=["box"] * len(locations),
        geom_sizes=sizes,
        geom_locations=locations,
        geom_quats=quats,
        geom_names=names,
        geom_rgbas=rgbas,
        geom_frictions=[[1.2, 0.005, 0.0001]] * len(locations),
        density=170.0,
        locations_relative_to_center=True,
        joints="default",
        sites=sites,
        duplicate_collision_geoms=True,
    )


def _triangle_post(name: str = "triangle_post") -> CompositeObject:
    """Approximate a solid triangular prism with overlapping horizontal slices."""

    triangle_height = POST_SIDE * math.sqrt(3.0) / 2.0
    slices = 7
    slice_height = triangle_height / slices
    locations = []
    sizes = []
    for index in range(slices):
        fraction = (index + 0.5) / slices
        width = max(POST_SIDE * fraction, 0.012)
        y = -2.0 * triangle_height / 3.0 + (index + 0.5) * slice_height
        locations.append([0.0, y, 0.0])
        sizes.append([width / 2.0, slice_height * 0.55, POST_HEIGHT / 2.0])
    return CompositeObject(
        name=name,
        total_size=[POST_SIDE / 2.0, triangle_height * 2.0 / 3.0, POST_HEIGHT / 2.0],
        geom_types=["box"] * slices,
        geom_sizes=sizes,
        geom_locations=locations,
        geom_names=[f"slice_{index}" for index in range(slices)],
        geom_rgbas=[[0.14, 0.36, 0.88, 1.0]] * slices,
        geom_frictions=[[1.0, 0.005, 0.0001]] * slices,
        density=800.0,
        locations_relative_to_center=True,
        joints=None,
        duplicate_collision_geoms=True,
    )


class ThreeArmTriangleInsertion(ManipulationEnv):
    """Lift one triangular frame by three real grasps and place it around a post."""

    preview_only = True
    interaction_model = "real_mujoco_contact_v1"

    def __init__(
        self,
        robots=("Panda",) * 3,
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
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        global_image_size=512,
        wrist_image_size=256,
        seed=None,
        spawn_side: str | None = None,
    ):
        if spawn_side not in (None, *FRAME_INITIAL_X_RANGES):
            raise ValueError("spawn_side must be one of None, 'left', or 'right'")
        self.task = get_role_task("t1_triangle_frame")
        self.table_full_size = np.asarray(TABLE_FULL_SIZE, dtype=float)
        self.table_offset = np.array([0.0, 0.0, TABLE_HEIGHT])
        self.table_friction = (1.0, 0.005, 0.0001)
        self.use_object_obs = use_object_obs
        self.frame_side = FRAME_SIDE
        self.frame_inner_side = FRAME_INNER_SIDE
        self.frame_height = FRAME_HEIGHT
        self.post_side = POST_SIDE
        self.post_height = POST_HEIGHT
        self.post_xy = np.array([POST_X, 0.0], dtype=np.float64)
        self.scene_offset = np.zeros(2, dtype=np.float64)
        self.scene_yaw = 0.0
        self.sampled_frame_pose = np.zeros(7, dtype=np.float64)
        self.requested_spawn_side = spawn_side
        self.spawn_side = ""
        self.spawn_sampling_attempts = 0
        self._reset_task_history()

        cameras = (*GLOBAL_CAMERAS, *LOCAL_CAMERAS)
        camera_sizes = [global_image_size] * 2 + [wrist_image_size] * 3
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
        for robot, (x, y) in zip(self.robots, THREE_ARM_BASE_XY):
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
        arena.set_camera(
            "frontview",
            pos=front_position,
            quat=_look_at_quat(front_position),
        )
        arena.worldbody.find(".//camera[@name='frontview']").set("fovy", "52")

        self.frame = _triangle_frame()
        self.post = _triangle_post()
        self.post._obj.set(
            "pos", array_to_string([*self.post_xy, TABLE_HEIGHT + POST_HEIGHT / 2.0])
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.frame, self.post],
        )

    def _setup_references(self):
        super()._setup_references()
        self.frame_body_id = self.sim.model.body_name2id(self.frame.root_body)
        self.handle_site_ids = OrderedDict(
            (
                role,
                self.sim.model.site_name2id(f"{self.frame.naming_prefix}handle_{role}_site"),
            )
            for role in HANDLE_ROLES
        )
        frame_geoms = {self.sim.model.geom_name2id(name) for name in self.frame.contact_geoms}
        post_geoms = {self.sim.model.geom_name2id(name) for name in self.post.contact_geoms}
        robot_geoms = {
            self.sim.model.geom_name2id(name)
            for robot in self.robots
            for name in robot.robot_model.contact_geoms
        }
        self._forbidden_contact_pairs = {
            frozenset((first, second))
            for first, others in ((frame_geoms, post_geoms | robot_geoms), (post_geoms, robot_geoms))
            for first in first
            for second in others
        }

    def _reset_internal(self):
        super()._reset_internal()
        self.spawn_side = self.requested_spawn_side or (
            "left" if self.rng.uniform() < 0.5 else "right"
        )
        x_range = FRAME_INITIAL_X_RANGES[self.spawn_side]
        joint_id = self.sim.model.joint_name2id(self.frame.joints[0])
        dof_start = int(self.sim.model.jnt_dofadr[joint_id])
        for attempt in range(1, 101):
            position = np.array(
                [
                    self.rng.uniform(*x_range),
                    self.rng.uniform(*FRAME_INITIAL_Y_RANGE),
                    TABLE_HEIGHT + FRAME_HEIGHT / 2.0 + 0.003,
                ],
                dtype=np.float64,
            )
            yaw = float(
                self.rng.uniform(-FRAME_INITIAL_YAW_RANGE, FRAME_INITIAL_YAW_RANGE)
            )
            quat = _yaw_quat(yaw)
            self.sim.data.set_joint_qpos(
                self.frame.joints[0], np.concatenate([position, quat])
            )
            self.sim.data.qvel[dof_start : dof_start + 6] = 0.0
            self.sim.forward()
            if self.initial_forbidden_contacts() == 0:
                self.spawn_sampling_attempts = attempt
                break
        else:
            raise RuntimeError("failed to sample a collision-free T1 frame pose")
        self.sampled_frame_pose = np.concatenate([position, quat]).copy()
        nominal_x = 0.5 * (x_range[0] + x_range[1])
        self.scene_offset = position[:2] - np.array([nominal_x, 0.0])
        self.scene_yaw = yaw
        self._reset_task_history()
        self.sim.forward()

    def frame_pose(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.sim.data.body_xpos[self.frame_body_id]),
                np.asarray(self.sim.data.body_xquat[self.frame_body_id]),
            ]
        ).copy()

    def handle_positions(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict(
            (role, np.asarray(self.sim.data.site_xpos[site_id]).copy())
            for role, site_id in self.handle_site_ids.items()
        )

    def handle_offsets(self) -> np.ndarray:
        pose = self.frame_pose()
        rotation = T.quat2mat(T.convert_quat(pose[3:], to="xyzw"))
        return np.stack(
            [rotation.T @ (position - pose[:3]) for position in self.handle_positions().values()]
        )

    def assigned_grasps(self) -> tuple[bool, bool, bool]:
        result = []
        for robot, role in zip(self.robots, HANDLE_ROLES):
            handle_geom = f"{self.frame.naming_prefix}handle_{role}"
            # The Panda pad geoms cover only the innermost fingertip faces. During
            # cooperative transport a handle can remain physically pinched between
            # both finger links while one tiny pad geom momentarily loses contact.
            # Measure grasp contact across both complete finger links to include
            # these physically supported transport configurations.
            arm_grasps = []
            for gripper in robot.gripper.values():
                finger_groups = (
                    gripper.important_geoms["left_finger"],
                    gripper.important_geoms["right_finger"],
                )
                arm_grasps.append(self._check_grasp(finger_groups, handle_geom))
            result.append(any(arm_grasps))
        return tuple(result)

    def initial_forbidden_contacts(self) -> int:
        return sum(
            frozenset((int(contact.geom1), int(contact.geom2))) in self._forbidden_contact_pairs
            for contact in self.sim.data.contact[: self.sim.data.ncon]
        )

    def _reset_task_history(self) -> None:
        self._history = {name: False for name in TRIANGLE_SUCCESS_EVIDENCE}
        self._grasp_steps = 0
        self._stable_steps = 0
        self._success_latched = False
        self._last_task_update_time = float("-inf")
        self._latest_task_state = {}

    def _task_state(self) -> dict[str, object]:
        now = float(self.sim.data.time)
        if now == self._last_task_update_time and self._latest_task_state:
            return dict(self._latest_task_state)

        pose = self.frame_pose()
        rotation = T.quat2mat(T.convert_quat(pose[3:], to="xyzw"))
        tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
        bottom_height = float(pose[2] - FRAME_HEIGHT / 2.0)
        center_error = float(np.linalg.norm(pose[:2] - self.post_xy))
        grasps = self.assigned_grasps()

        self._grasp_steps = self._grasp_steps + 1 if all(grasps) else 0
        if self._grasp_steps >= GRASP_CONFIRM_STEPS:
            self._history["three_grasp_confirmed"] = True
        if (
            self._history["three_grasp_confirmed"]
            and bottom_height >= TABLE_HEIGHT + POST_HEIGHT + 0.025
        ):
            self._history["lifted"] = True
        if self._history["lifted"] and center_error <= 0.05:
            self._history["transported"] = True

        placed_now = bool(
            center_error <= CENTER_TOLERANCE
            and tilt <= TILT_TOLERANCE_RADIANS
            and TABLE_HEIGHT - 0.003
            <= bottom_height
            <= TABLE_HEIGHT + PLACEMENT_HEIGHT_TOLERANCE
        )
        if self._history["transported"] and placed_now:
            self._history["placed"] = True
        if self._history["placed"] and not any(grasps):
            self._history["released"] = True

        retreat_distances = tuple(
            float(np.linalg.norm(
                np.asarray(self.sim.data.site_xpos[robot.eef_site_id["right"]]) - handle
            ))
            for robot, handle in zip(self.robots, self.handle_positions().values())
        )
        if self._history["released"] and min(retreat_distances) >= RETREAT_DISTANCE:
            self._history["retreated"] = True

        linear_speed = float(np.linalg.norm(self.sim.data.get_body_xvelp(self.frame.root_body)))
        angular_speed = float(np.linalg.norm(self.sim.data.get_body_xvelr(self.frame.root_body)))
        stable_now = bool(
            self._history["retreated"]
            and placed_now
            and linear_speed <= STABLE_LINEAR_SPEED
            and angular_speed <= STABLE_ANGULAR_SPEED
        )
        self._stable_steps = self._stable_steps + 1 if stable_now else 0
        if self._stable_steps >= STABLE_STEPS:
            self._history["stable"] = True
        self._success_latched |= triangle_success_from_state(self._history)

        state = {
            "task_id": self.task.task_id,
            "preview_only": True,
            "interaction_model": self.interaction_model,
            "scene_offset": self.scene_offset.copy(),
            "scene_yaw": self.scene_yaw,
            "spawn_side": self.spawn_side,
            "spawn_sampling_attempts": self.spawn_sampling_attempts,
            "frame_position": pose[:3].copy(),
            "sampled_frame_pose": self.sampled_frame_pose.copy(),
            "start_center_separation": float(
                np.linalg.norm(self.sampled_frame_pose[:2] - self.post_xy)
            ),
            "initial_forbidden_contacts": self.initial_forbidden_contacts(),
            "assigned_grasps": grasps,
            "grasp_confirm_steps": self._grasp_steps,
            "center_error": center_error,
            "tilt_degrees": math.degrees(tilt),
            "frame_bottom_height": bottom_height,
            "retreat_distances": retreat_distances,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
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
        if state["success"]:
            return 1.0
        return 0.05 * sum(state["assigned_grasps"])


def make_triangle_frame_insertion_env(
    *,
    global_image_size: int = 512,
    wrist_image_size: int = 256,
    horizon: int = 1000,
    seed: int | None = None,
    spawn_side: str | None = None,
):
    return ThreeArmTriangleInsertion(
        global_image_size=global_image_size,
        wrist_image_size=wrist_image_size,
        horizon=horizon,
        seed=seed,
        spawn_side=spawn_side,
    )


__all__ = (
    "GLOBAL_CAMERAS",
    "HANDLE_ROLES",
    "LOCAL_CAMERAS",
    "ThreeArmTriangleInsertion",
    "make_triangle_frame_insertion_env",
    "triangle_success_from_state",
)
