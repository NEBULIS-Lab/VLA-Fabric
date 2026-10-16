"""Four-Panda cooperative square-frame placement around a fixed post."""

from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CompositeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor

from multiarm_sim.frame_control import success_from_state


GLOBAL_CAMERA = "agentview"
LOCAL_CAMERAS = tuple(f"robot{i}_eye_in_hand" for i in range(4))
CONTROL_FREQUENCY = 20
HANDLE_ROLES = ("northwest", "northeast", "southeast", "southwest")

TABLE_FULL_SIZE = (1.60, 1.60, 0.05)
TABLE_HEIGHT = 0.80
ROBOT_BASE_XY = (
    (-0.56, 0.56),
    (0.56, 0.56),
    (0.56, -0.56),
    (-0.56, -0.56),
)
FRAME_SPAWN_X = (-0.32, 0.32)
GRASP_CONFIRM_STEPS = 3
CENTER_TOLERANCE = 0.02
TILT_TOLERANCE_RADIANS = math.radians(5.0)
PLACEMENT_HEIGHT_TOLERANCE = 0.008
RETREAT_DISTANCE = 0.04
STABLE_LINEAR_SPEED = 0.02
STABLE_ANGULAR_SPEED = 0.10
STABLE_STEPS = 10


def _square_frame(name: str = "square_frame") -> CompositeObject:
    outer = 0.38
    inner = 0.24
    height = 0.045
    rail = (outer - inner) / 2
    handle_length = 0.12
    handle_width = 0.04
    half_outer = outer / 2
    half_inner = inner / 2
    rail_center = half_inner + rail / 2
    diagonal_offset = handle_length / (2 * math.sqrt(2))

    locations = [
        [0.0, rail_center, 0.0],
        [0.0, -rail_center, 0.0],
        [-rail_center, 0.0, 0.0],
        [rail_center, 0.0, 0.0],
    ]
    sizes = [
        [half_outer, rail / 2, height / 2],
        [half_outer, rail / 2, height / 2],
        [rail / 2, half_inner, height / 2],
        [rail / 2, half_inner, height / 2],
    ]
    names = ["north_rail", "south_rail", "west_rail", "east_rail"]
    quats = [None] * 4
    rgbas = [[0.95, 0.48, 0.08, 1.0]] * 4

    handle_angles = (3 * math.pi / 4, math.pi / 4, -math.pi / 4, -3 * math.pi / 4)
    handle_signs = ((-1, 1), (1, 1), (1, -1), (-1, -1))
    role_colors = (
        [0.90, 0.15, 0.12, 1.0],
        [0.15, 0.70, 0.22, 1.0],
        [0.15, 0.34, 0.92, 1.0],
        [0.92, 0.78, 0.12, 1.0],
    )
    sites = []
    for role, angle, (sign_x, sign_y), color in zip(
        HANDLE_ROLES, handle_angles, handle_signs, role_colors
    ):
        center = [
            sign_x * (half_outer + diagonal_offset),
            sign_y * (half_outer + diagonal_offset),
            0.0,
        ]
        locations.append(center)
        sizes.append([handle_length / 2, handle_width / 2, handle_width / 2])
        names.append(f"handle_{role}")
        quats.append(T.convert_quat(T.axisangle2quat(np.array([0.0, 0.0, angle])), to="wxyz"))
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

    bound = half_outer + handle_length / math.sqrt(2) + 0.01
    return CompositeObject(
        name=name,
        total_size=[bound, bound, height / 2],
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


class FourArmFrameInsertion(ManipulationEnv):
    """Place one four-handle hollow frame around a fixed square post."""

    def __init__(
        self,
        robots=("Panda",) * 4,
        controller_configs=None,
        base_types="NullMount",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=GLOBAL_CAMERA,
        render_gpu_device_id=-1,
        control_freq=CONTROL_FREQUENCY,
        horizon=1600,
        ignore_done=True,
        hard_reset=True,
        global_image_size=512,
        wrist_image_size=256,
        seed=None,
    ):
        self.table_full_size = np.asarray(TABLE_FULL_SIZE, dtype=float)
        self.table_offset = np.array([0.0, 0.0, TABLE_HEIGHT])
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.frame_outer_size = 0.38
        self.frame_inner_size = 0.24
        self.frame_height = 0.045
        self.post_width = 0.16
        self.post_height = 0.18
        self.spawn_side = "left"
        self.sampled_frame_pose = np.zeros(7, dtype=np.float64)
        self._reset_task_history()

        cameras = (GLOBAL_CAMERA, *LOCAL_CAMERAS)
        camera_sizes = [global_image_size] + [wrist_image_size] * 4
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
            camera_depths=[False] * 5,
            camera_segmentations=[None] * 5,
            seed=seed,
        )

    @property
    def instruction(self) -> str:
        return (
            "Use all four arms to lift the square frame, place it around the center post, "
            "release every handle, and retreat."
        )

    def _load_model(self):
        super()._load_model()

        for robot, (x, y) in zip(self.robots, ROBOT_BASE_XY):
            yaw = math.atan2(-y, -x)
            robot.robot_model.set_base_xpos([x, y, TABLE_HEIGHT])
            robot.robot_model.set_base_ori([0.0, 0.0, yaw])

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0.0, 0.0, 0.0])
        arena.set_camera(
            camera_name=GLOBAL_CAMERA,
            pos=[0.0, 0.0, 2.65],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        arena.worldbody.find(f".//camera[@name='{GLOBAL_CAMERA}']").set("fovy", "48")

        self.frame = _square_frame()
        self.post = BoxObject(
            name="square_post",
            size=[self.post_width / 2, self.post_width / 2, self.post_height / 2],
            rgba=[0.16, 0.38, 0.85, 1.0],
            joints=None,
            density=800.0,
            friction=[1.0, 0.005, 0.0001],
            rng=self.rng,
        )
        self.post._obj.set(
            "pos",
            array_to_string([0.0, 0.0, TABLE_HEIGHT + self.post_height / 2]),
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
            frozenset((frame_geom, other_geom))
            for frame_geom in frame_geoms
            for other_geom in post_geoms | robot_geoms
        }

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        @sensor(modality="object")
        def frame_pos(obs_cache):
            return np.asarray(self.sim.data.body_xpos[self.frame_body_id]).copy()

        @sensor(modality="object")
        def frame_quat(obs_cache):
            return np.asarray(self.sim.data.body_xquat[self.frame_body_id]).copy()

        observables[frame_pos.__name__] = Observable(
            name=frame_pos.__name__, sensor=frame_pos, sampling_rate=self.control_freq
        )
        observables[frame_quat.__name__] = Observable(
            name=frame_quat.__name__, sensor=frame_quat, sampling_rate=self.control_freq
        )
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        side_index = int(self.rng.integers(0, 2))
        self.spawn_side = "left" if side_index == 0 else "right"
        position = np.array(
            [
                FRAME_SPAWN_X[side_index] + self.rng.uniform(-0.02, 0.02),
                self.rng.uniform(-0.03, 0.03),
                TABLE_HEIGHT + self.frame_height / 2 + 0.002,
            ],
            dtype=np.float64,
        )
        yaw = self.rng.uniform(-math.radians(5.0), math.radians(5.0))
        quat_xyzw = T.axisangle2quat(np.array([0.0, 0.0, yaw]))
        quat_wxyz = T.convert_quat(quat_xyzw, to="wxyz")
        self.sim.data.set_joint_qpos(
            self.frame.joints[0], np.concatenate([position, quat_wxyz])
        )
        self.sampled_frame_pose = np.concatenate([position, quat_wxyz]).copy()
        self._reset_task_history()

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

    def initial_forbidden_contacts(self) -> int:
        return sum(
            frozenset((int(contact.geom1), int(contact.geom2)))
            in self._forbidden_contact_pairs
            for contact in self.sim.data.contact[: self.sim.data.ncon]
        )

    def _reset_task_history(self) -> None:
        self._history = {
            "four_grasp_confirmed": False,
            "lifted": False,
            "transported": False,
            "placed": False,
            "released": False,
            "retreated": False,
            "stable": False,
        }
        self._grasp_steps = 0
        self._stable_steps = 0
        self._success_latched = False
        self._last_task_update_time = float("-inf")
        self._latest_task_state = {}

    def assigned_grasps(self) -> tuple[bool, bool, bool, bool]:
        """Whether each Panda grips its matching colored corner handle."""

        result = []
        for robot, role in zip(self.robots, HANDLE_ROLES):
            handle_geom = f"{self.frame.naming_prefix}handle_{role}"
            result.append(bool(self._check_grasp(robot.gripper, handle_geom)))
        return tuple(result)

    def handle_offsets(self) -> np.ndarray:
        """Return the four handle centers in the rigid frame coordinate system."""

        pose = self.frame_pose()
        rotation = T.quat2mat(T.convert_quat(pose[3:], to="xyzw"))
        return np.stack(
            [rotation.T @ (position - pose[:3]) for position in self.handle_positions().values()]
        )

    def _task_state(self) -> dict[str, object]:
        now = float(self.sim.data.time)
        if now == self._last_task_update_time and self._latest_task_state:
            return dict(self._latest_task_state)

        pose = self.frame_pose()
        frame_rotation = T.quat2mat(T.convert_quat(pose[3:], to="xyzw"))
        tilt = math.acos(float(np.clip(frame_rotation[2, 2], -1.0, 1.0)))
        bottom_height = float(pose[2] - self.frame_height / 2)
        center_error = float(np.linalg.norm(pose[:2]))
        grasps = self.assigned_grasps()

        self._grasp_steps = self._grasp_steps + 1 if all(grasps) else 0
        if self._grasp_steps >= GRASP_CONFIRM_STEPS:
            self._history["four_grasp_confirmed"] = True
        if (
            self._history["four_grasp_confirmed"]
            and bottom_height >= TABLE_HEIGHT + self.post_height + 0.03
        ):
            self._history["lifted"] = True
        if self._history["lifted"] and center_error <= 0.06:
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

        eef_positions = [
            np.asarray(self.sim.data.site_xpos[robot.eef_site_id["right"]]).copy()
            for robot in self.robots
        ]
        retreat_distances = tuple(
            float(np.linalg.norm(eef - handle))
            for eef, handle in zip(eef_positions, self.handle_positions().values())
        )
        if self._history["released"] and min(retreat_distances) >= RETREAT_DISTANCE:
            self._history["retreated"] = True

        linear_speed = float(
            np.linalg.norm(self.sim.data.get_body_xvelp(self.frame.root_body))
        )
        angular_speed = float(
            np.linalg.norm(self.sim.data.get_body_xvelr(self.frame.root_body))
        )
        stable_now = bool(
            self._history["retreated"]
            and placed_now
            and linear_speed <= STABLE_LINEAR_SPEED
            and angular_speed <= STABLE_ANGULAR_SPEED
        )
        self._stable_steps = self._stable_steps + 1 if stable_now else 0
        if self._stable_steps >= STABLE_STEPS:
            self._history["stable"] = True
        self._success_latched |= success_from_state(self._history)

        state = {
            "spawn_side": self.spawn_side,
            "frame_position": pose[:3].copy(),
            "sampled_frame_pose": self.sampled_frame_pose.copy(),
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
            reward = 1.0
        elif state["retreated"]:
            reward = 0.95
        elif state["released"]:
            reward = 0.90
        elif state["placed"]:
            reward = 0.82
        elif state["transported"]:
            reward = 0.68
        elif state["lifted"]:
            reward = 0.48
        elif state["four_grasp_confirmed"]:
            reward = 0.30
        else:
            reward = 0.05 * sum(state["assigned_grasps"])
        return reward if self.reward_scale is None else reward * self.reward_scale


def make_frame_insertion_env(
    *,
    global_image_size: int = 512,
    wrist_image_size: int = 256,
    horizon: int = 1600,
    seed: int | None = None,
):
    """Construct the four-Panda frame task used by collection and replay."""
    if seed is not None:
        np.random.seed(seed)
    return FourArmFrameInsertion(
        robots=("Panda",) * 4,
        global_image_size=global_image_size,
        wrist_image_size=wrist_image_size,
        horizon=horizon,
        seed=seed,
    )


def top_left_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(image))


def frame_from_observation(observation: dict, camera: str) -> np.ndarray:
    return top_left_rgb(observation[f"{camera}_image"])


def proprio_from_observation(observation: dict, robot_index: int) -> np.ndarray:
    return np.concatenate(
        [
            observation[f"robot{robot_index}_joint_pos"],
            observation[f"robot{robot_index}_gripper_qpos"],
        ]
    ).astype(np.float32)
