"""Short closed-loop real-contact expert for four-arm arch assembly."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import robosuite.utils.transform_utils as T

from multiarm_sim.envs.arch_assembly import FOUR_ARM_BASE_XY
from multiarm_sim.frame_control import world_delta_to_base


PHASES = (
    "approach", "descend", "close", "lift", "align", "place_pillars",
    "handoff_approach", "handoff_close", "dual_transport",
    "dual_seat_beam", "dual_release", "retreat", "settle",
)
PHASE_TIMEOUTS = (140, 100, 100, 240, 240, 180, 280, 100, 400, 400, 80, 140, 160)
RECEIVER_GRASP_Z_OFFSET = 0.0
RECEIVER_GRASP_OFFSET = np.array([0.0, 0.0, RECEIVER_GRASP_Z_OFFSET])


def eef_position_actions(
    observation: Mapping[str, np.ndarray], targets: np.ndarray, *,
    base_yaws, grippers, orientation_references=None, yaw_offsets=None,
    gain: float = 8.0, limit: float = 0.6,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (4, 3):
        raise ValueError(f"targets must have shape (4, 3), got {targets.shape}")
    actions = np.zeros((4, 7), dtype=np.float64)
    for index, (target, yaw, closed) in enumerate(zip(targets, base_yaws, grippers)):
        current = np.asarray(observation[f"robot{index}_eef_pos"], dtype=np.float64)
        local = world_delta_to_base(gain * (target - current), float(yaw))
        actions[index, :3] = np.clip(local, -limit, limit)
        if orientation_references is not None and yaw_offsets is not None:
            current_mat = T.quat2mat(
                np.asarray(observation[f"robot{index}_eef_quat"], dtype=np.float64)
            )
            relative_mat = current_mat @ orientation_references[index].T
            current_yaw = math.atan2(relative_mat[1, 0], relative_mat[0, 0])
            yaw_error = (float(yaw_offsets[index]) - current_yaw + math.pi) % (
                2.0 * math.pi
            ) - math.pi
            # OSC maps a unit rotation action to 0.5 rad.
            actions[index, 5] = np.clip(2.0 * yaw_error, -1.0, 1.0)
        actions[index, 6] = 1.0 if closed else -1.0
    return actions.reshape(-1)


class ScriptedArchExpert:
    """Build a small arch using physical contacts and one beam handoff."""

    def __init__(self, env) -> None:
        self.env = env
        self.base_yaws = tuple(math.atan2(-y, -x) for x, y in FOUR_ARM_BASE_XY)
        self.phase_index = 0
        self.phase_steps = 0
        self.ready_steps = 0
        self.done = False
        self.failed = False
        self.failure_reason = ""
        self.orientation_references = None
        self.yaw_offsets = (0.0, 0.0, math.pi / 4.0, -math.pi / 4.0)

        poses = env.object_poses()
        sites = env.grasp_sites()
        self.supplier_offset = sites["beam_supplier"] - poses["beam"][:3]
        self.receiver_offset = sites["beam_receiver"] - poses["beam"][:3]
        self.receiver_wait = env.beam_handoff_target + self.receiver_offset
        self.receiver_wait[2] += 0.10
        self.home_targets = None
        self.pre_approach_targets = None
        self.close_hold_targets = None
        self.handoff_hold_targets = None
        self.upper_release_targets = None
        self.upper_clear_targets = None
        self.upper_return_stage = 0
        self.lift_root_targets = None
        self.transport_eef_offsets = None
        self.dual_eef_offsets = None
        self.dual_stage = 0
        self.dual_release_targets = None
        self.safe_retreat_targets = None

    @property
    def phase(self) -> str:
        return PHASES[self.phase_index]

    @property
    def stage(self) -> int:
        return self.phase_index

    @property
    def next_progress(self) -> float:
        return min(self.phase_steps / PHASE_TIMEOUTS[self.phase_index], 1.0)

    def _advance(self) -> None:
        self.phase_index = min(self.phase_index + 1, len(PHASES) - 1)
        self.phase_steps = 0
        self.ready_steps = 0

    def _advance_when(self, ready: bool, *, hold: int, timeout: int) -> None:
        self.ready_steps = self.ready_steps + 1 if ready else 0
        if self.ready_steps >= hold:
            self._advance()
        elif self.phase_steps >= timeout:
            self.failed = True
            self.done = True
            self.failure_reason = f"{self.phase} timeout"

    def _command(self, observation, targets, grippers, *, gain=8.0, limit=0.6):
        return eef_position_actions(
            observation, targets, base_yaws=self.base_yaws,
            grippers=grippers,
            orientation_references=self.orientation_references,
            yaw_offsets=self.yaw_offsets,
            gain=gain, limit=limit,
        )

    def action(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.done:
            idle = np.zeros((4, 7), dtype=np.float32)
            idle[:, 6] = -1.0
            return idle.reshape(-1)

        self.phase_steps += 1
        grasp_sites = np.stack(list(self.env.grasp_sites().values()))
        eefs = np.stack([
            np.asarray(observation[f"robot{i}_eef_pos"], dtype=np.float64)
            for i in range(4)
        ])
        poses = self.env.object_poses()
        status = self.env.task_status()

        if self.orientation_references is None:
            self.orientation_references = tuple(
                T.quat2mat(
                    np.asarray(observation[f"robot{i}_eef_quat"], dtype=np.float64)
                )
                for i in range(4)
            )

        if self.pre_approach_targets is None:
            self.home_targets = eefs.copy()
            self.pre_approach_targets = eefs.copy()
            self.pre_approach_targets[:, 2] += 0.14
        if self.phase == "handoff_approach" and self.upper_release_targets is None:
            self.upper_release_targets = eefs[:2].copy()
            self.upper_clear_targets = self.upper_release_targets + np.array([0.0, 0.0, 0.10])
        if self.phase == "close" and self.close_hold_targets is None:
            self.close_hold_targets = eefs.copy()
            self.close_hold_targets[:2, 2] -= 0.012
            # The long beam is thinner than the upright pillars; a small
            # additional descent puts both finger pads on its body without
            # continuously driving the wrist into the table.
            self.close_hold_targets[2, 2] -= 0.014
        if self.phase == "handoff_close" and self.handoff_hold_targets is None:
            self.handoff_hold_targets = eefs.copy()
            self.handoff_hold_targets[3, 2] -= 0.012
        if self.phase == "lift" and self.lift_root_targets is None:
            self.lift_root_targets = np.stack([
                poses["left_pillar"][:3] + np.array([0.0, 0.0, 0.14]),
                poses["right_pillar"][:3] + np.array([0.0, 0.0, 0.14]),
                poses["beam"][:3] + np.array([0.0, 0.0, 0.17]),
            ])
            self.transport_eef_offsets = eefs[:3] - np.stack([
                poses["left_pillar"][:3], poses["right_pillar"][:3],
                poses["beam"][:3],
            ])
        if self.phase == "dual_transport" and self.dual_eef_offsets is None:
            self.dual_eef_offsets = eefs[2:4] - poses["beam"][:3]
        if self.phase == "dual_release" and self.dual_release_targets is None:
            self.dual_release_targets = eefs.copy()
        if self.phase == "retreat" and self.safe_retreat_targets is None:
            self.safe_retreat_targets = eefs.copy()
            self.safe_retreat_targets[:, 2] += 0.12

        if self.phase == "approach":
            if self.phase_steps <= 32:
                targets = self.pre_approach_targets
            else:
                targets = grasp_sites.copy()
                targets[:3, 2] += 0.08
                targets[3] = self.receiver_wait
            action = self._command(observation, targets, (False,) * 4)
            ready = self.phase_steps > 32 and np.max(
                np.linalg.norm(targets[:3] - eefs[:3], axis=1)
            ) < 0.025
            self._advance_when(bool(ready), hold=4, timeout=140)

        elif self.phase == "descend":
            targets = grasp_sites.copy()
            targets[:3, 2] += 0.003
            targets[3] = self.receiver_wait
            action = self._command(observation, targets, (False,) * 4)
            # The Panda EEF site sits above the finger pads, so direct body
            # contact stops it roughly 4 cm short of the block centre.
            ready = np.max(np.linalg.norm(targets[:3] - eefs[:3], axis=1)) < 0.055
            self._advance_when(bool(ready), hold=6, timeout=100)

        elif self.phase == "close":
            targets = self.close_hold_targets
            action = self._command(observation, targets, (True, True, True, False))
            if status["initial_grasps_confirmed"] and self.phase_steps >= 20:
                self._advance()
            elif self.phase_steps >= 100:
                self.failed = True
                self.done = True
                self.failure_reason = f"initial grasp timeout: {status['assigned_grasps']}"

        elif self.phase == "lift":
            targets = np.stack([
                self.lift_root_targets[0] + self.transport_eef_offsets[0],
                self.lift_root_targets[1] + self.transport_eef_offsets[1],
                self.lift_root_targets[2] + self.transport_eef_offsets[2],
                self.receiver_wait,
            ])
            action = self._command(
                observation, targets, (True, True, True, False),
                gain=4.0, limit=0.20,
            )
            ready = (
                status["pillars_lifted"]
                and min(status["assigned_grasps"][:3])
                and np.min(
                    np.stack([
                        poses["left_pillar"][:3], poses["right_pillar"][:3],
                        poses["beam"][:3],
                    ])[:, 2] - self.lift_root_targets[:, 2]
                ) >= -0.055
            )
            self._advance_when(bool(ready), hold=4, timeout=240)

        elif self.phase == "align":
            pillar_roots = self.env.pillar_targets.copy()
            pillar_roots[:, 2] += 0.10
            targets = np.stack([
                pillar_roots[0] + self.transport_eef_offsets[0],
                pillar_roots[1] + self.transport_eef_offsets[1],
                self.env.beam_handoff_target + self.transport_eef_offsets[2],
                self.receiver_wait,
            ])
            action = self._command(
                observation, targets, (True, True, True, False),
                gain=4.0, limit=0.24,
            )
            pillar_xy = np.stack([poses["left_pillar"][:2], poses["right_pillar"][:2]])
            ready = (
                np.max(np.linalg.norm(pillar_xy - self.env.pillar_targets[:, :2], axis=1)) < 0.035
                and np.linalg.norm(poses["beam"][:3] - self.env.beam_handoff_target) < 0.055
            )
            self._advance_when(bool(ready), hold=4, timeout=240)

        elif self.phase == "place_pillars":
            targets = np.stack([
                self.env.pillar_targets[0] + self.transport_eef_offsets[0],
                self.env.pillar_targets[1] + self.transport_eef_offsets[1],
                self.env.beam_handoff_target + self.transport_eef_offsets[2],
                self.receiver_wait,
            ])
            action = self._command(observation, targets, (True, True, True, False))
            pillar_errors = np.linalg.norm(
                np.stack([
                    poses["left_pillar"][:3], poses["right_pillar"][:3]
                ]) - self.env.pillar_targets,
                axis=1,
            )
            ready = status["pillars_placed"] and np.max(pillar_errors) < 0.020
            self._advance_when(bool(ready), hold=6, timeout=180)

        elif self.phase == "handoff_approach":
            # Release the pillars, lift vertically clear of the arch, and
            # return both upper arms to their initial neutral poses. Robot 3
            # is kept waiting until that complete sequence has finished.
            if self.upper_return_stage == 0:
                upper_targets = self.upper_release_targets
                if self.phase_steps >= 14 and not any(status["assigned_grasps"][:2]):
                    self.upper_return_stage = 1
            elif self.upper_return_stage == 1:
                upper_targets = self.upper_clear_targets
                if np.max(np.linalg.norm(upper_targets - eefs[:2], axis=1)) < 0.035:
                    self.upper_return_stage = 2
            else:
                upper_targets = self.home_targets[:2]
                if np.max(np.linalg.norm(upper_targets - eefs[:2], axis=1)) < 0.035:
                    self.upper_return_stage = 3
            receiver_target = (
                grasp_sites[3] + RECEIVER_GRASP_OFFSET
                if self.upper_return_stage >= 3
                else self.receiver_wait
            )
            targets = np.stack([
                upper_targets[0], upper_targets[1],
                self.env.beam_handoff_target + self.transport_eef_offsets[2],
                receiver_target,
            ])
            action = self._command(observation, targets, (False, False, True, False))
            ready = (
                self.upper_return_stage >= 3
                and not status["assigned_grasps"][0]
                and not status["assigned_grasps"][1]
                and np.linalg.norm(targets[3] - eefs[3]) < 0.055
            )
            self._advance_when(bool(ready), hold=5, timeout=280)

        elif self.phase == "handoff_close":
            targets = np.stack([
                self.home_targets[0], self.home_targets[1],
                self.handoff_hold_targets[2],
                self.handoff_hold_targets[3],
            ])
            action = self._command(observation, targets, (False, False, True, True))
            if status["beam_handoff_confirmed"] and self.phase_steps >= 12:
                self._advance()
            elif self.phase_steps >= 100:
                self.failed = True
                self.done = True
                self.failure_reason = f"beam handoff timeout: {status['assigned_grasps']}"

        elif self.phase == "dual_transport":
            safe_z = self.env.beam_target[2] + 0.10
            if self.dual_stage == 0:
                root_target = np.array([*self.env.beam_handoff_target[:2], safe_z])
                if poses["beam"][2] >= safe_z - 0.045:
                    self.dual_stage = 1
            else:
                root_target = np.array([*self.env.beam_target[:2], safe_z])
            beam_delta = root_target - poses["beam"][:3]
            targets = np.stack([
                self.home_targets[0], self.home_targets[1],
                eefs[2] + beam_delta,
                eefs[3] + beam_delta,
            ])
            action = self._command(
                observation, targets, (False, False, True, True),
                gain=4.0, limit=0.24,
            )
            grasps = status["assigned_grasps"]
            ready = (
                self.dual_stage == 1
                and grasps[2]
                and grasps[3]
                and poses["beam"][2] >= safe_z - 0.055
                and np.linalg.norm(
                    poses["beam"][:2] - self.env.beam_target[:2]
                ) < 0.045
            )
            self._advance_when(bool(ready), hold=3, timeout=400)

        elif self.phase == "dual_seat_beam":
            # Give both beam arms the same world-frame translation on every
            # step. Reapplying offsets frozen at handoff creates closed-chain
            # strain after tiny physical slips or rotations and can pull one
            # gripper off the beam before it reaches the pillar tops.
            root_target = self.env.beam_target
            beam_delta = root_target - poses["beam"][:3]
            targets = np.stack([
                self.home_targets[0], self.home_targets[1],
                eefs[2] + beam_delta,
                eefs[3] + beam_delta,
            ])
            action = self._command(
                observation, targets, (False, False, True, True),
                gain=2.4, limit=0.10,
            )
            xy_error = np.linalg.norm(
                poses["beam"][:2] - self.env.beam_target[:2]
            )
            ready = (
                status["beam_seated"]
                and status["beam_error"] < 0.028
                and xy_error < 0.025
            )
            self._advance_when(bool(ready), hold=3, timeout=400)

        elif self.phase == "dual_release":
            action = self._command(
                observation, self.dual_release_targets, (False,) * 4,
                gain=2.4, limit=0.10,
            )
            self._advance_when(
                bool(status["dual_release_after_seating"]), hold=12, timeout=80
            )

        elif self.phase == "retreat":
            action = self._command(observation, self.safe_retreat_targets, (False,) * 4)
            self._advance_when(bool(status["retreated"]), hold=4, timeout=140)

        else:
            action = np.zeros((4, 7), dtype=np.float64)
            action[:, 6] = -1.0
            if status["success"]:
                self.done = True
            elif self.phase_steps >= 160:
                self.failed = True
                self.done = True
                self.failure_reason = "stable success timeout"

        return np.asarray(action, dtype=np.float32).reshape(-1)


__all__ = ("PHASES", "ScriptedArchExpert", "eef_position_actions")
