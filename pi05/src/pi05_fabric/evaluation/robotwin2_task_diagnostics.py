"""Task-owned diagnostics, deliberately separate from official success logic."""

from __future__ import annotations


PHASE_FIELDS = ("block_lifted", "middle_reached", "target_region_reached")


def canonical_task_name(task_name: str) -> str:
    name = task_name.removeprefix("robotwin_")
    if name not in ("handover_block", "handover_mic", "hanging_mug", "scan_object"):
        raise ValueError(f"unsupported RoboTwin diagnostic task: {task_name}")
    return f"robotwin_{name}"


class BlockDiagnostics:
    task = "robotwin_handover_block"
    phase_metrics_status = "collected"
    initial_phases = (0, 0, 0)
    position_fields = ("box_position", "target_position")

    def start(self, environment):
        return float(environment.box.get_pose().p[2])

    def observe(self, environment, initial_z):
        import numpy as np

        block = np.asarray(environment.box.get_pose().p)
        target = np.asarray(environment.target_box.get_functional_point(1, "pose").p)
        lifted = int(block[2] > initial_z + 0.04)
        middle = int(abs(block[0]) < 0.08 and block[2] > 0.86)
        target_region = int(np.all(np.abs(block[:2] - target[:2]) < np.asarray([0.06, 0.06])))
        return lifted, middle, target_region

    def scene_state(self, environment):
        import numpy as np

        return {
            "box_position": np.asarray(environment.box.get_pose().p).tolist(),
            "target_position": np.asarray(environment.target_box.get_pose().p).tolist(),
        }

    def scene_repeatable(self, first, second):
        import numpy as np

        return all(np.allclose(first[key], second[key], atol=1e-7) for key in self.position_fields)


class MicDiagnostics:
    task = "robotwin_handover_mic"
    phase_metrics_status = "not_collected"
    initial_phases = (None, None, None)

    def start(self, environment):
        return None

    def observe(self, environment, initial_state):
        return self.initial_phases

    def scene_state(self, environment):
        import numpy as np

        pose = environment.microphone.get_pose()
        return {
            "microphone_position": np.asarray(pose.p).tolist(),
            "microphone_quaternion": np.asarray(pose.q).tolist(),
            "microphone_id": int(environment.microphone_id),
            "grasp_arm": str(environment.grasp_arm_tag),
            "handover_arm": str(environment.handover_arm_tag),
        }

    def scene_repeatable(self, first, second):
        import numpy as np

        geometry_matches = all(
            np.allclose(first[key], second[key], atol=1e-7, rtol=0)
            for key in ("microphone_position", "microphone_quaternion")
        )
        identity_matches = all(
            first[key] == second[key] for key in ("microphone_id", "grasp_arm", "handover_arm")
        )
        return geometry_matches and identity_matches


class HangingMugDiagnostics:
    task = "robotwin_hanging_mug"
    phase_metrics_status = "not_collected"
    initial_phases = (None, None, None)

    def start(self, environment):
        return None

    def observe(self, environment, initial_state):
        return self.initial_phases

    def scene_state(self, environment):
        import numpy as np

        mug_pose = environment.mug.get_pose()
        rack_pose = environment.rack.get_pose()
        return {
            "mug_position": np.asarray(mug_pose.p).tolist(),
            "mug_quaternion": np.asarray(mug_pose.q).tolist(),
            "mug_id": int(environment.mug_id),
            "rack_position": np.asarray(rack_pose.p).tolist(),
            "rack_quaternion": np.asarray(rack_pose.q).tolist(),
        }

    def scene_repeatable(self, first, second):
        import numpy as np

        geometry_matches = all(
            np.allclose(first[key], second[key], atol=1e-7, rtol=0)
            for key in (
                "mug_position",
                "mug_quaternion",
                "rack_position",
                "rack_quaternion",
            )
        )
        return geometry_matches and first["mug_id"] == second["mug_id"]


class ScanObjectDiagnostics:
    task = "robotwin_scan_object"
    phase_metrics_status = "collected"
    initial_phases = (0, 0, 0)
    phase_labels = (
        "both_grippers_closed",
        "scanner_and_object_lifted",
        "scan_pair_within_10cm",
    )

    def start(self, environment):
        return (
            float(environment.scanner.get_pose().p[2]),
            float(environment.object.get_pose().p[2]),
        )

    def observe(self, environment, initial_state):
        import numpy as np

        scanner_position = np.asarray(environment.scanner.get_pose().p)
        object_position = np.asarray(environment.object.get_pose().p)
        scanner_point = np.asarray(environment.scanner.get_functional_point(0))[:3]
        both_closed = int(
            environment.is_left_gripper_close()
            and environment.is_right_gripper_close()
        )
        both_lifted = int(
            scanner_position[2] > initial_state[0] + 0.04
            and object_position[2] > initial_state[1] + 0.04
        )
        scan_pair_near = int(np.linalg.norm(scanner_point - object_position) < 0.10)
        return both_closed, both_lifted, scan_pair_near

    def scene_state(self, environment):
        import numpy as np

        scanner_pose = environment.scanner.get_pose()
        object_pose = environment.object.get_pose()
        return {
            "scanner_position": np.asarray(scanner_pose.p).tolist(),
            "scanner_quaternion": np.asarray(scanner_pose.q).tolist(),
            "scanner_id": int(environment.scanner_id),
            "object_position": np.asarray(object_pose.p).tolist(),
            "object_quaternion": np.asarray(object_pose.q).tolist(),
            "object_id": int(environment.object_id),
        }

    def scene_repeatable(self, first, second):
        import numpy as np

        geometry = all(
            np.allclose(first[key], second[key], atol=1e-7, rtol=0)
            for key in (
                "scanner_position",
                "scanner_quaternion",
                "object_position",
                "object_quaternion",
            )
        )
        identity = all(
            first[key] == second[key] for key in ("scanner_id", "object_id")
        )
        return geometry and identity


def task_diagnostics(task_name: str):
    task = canonical_task_name(task_name)
    adapters = {
        BlockDiagnostics.task: BlockDiagnostics,
        MicDiagnostics.task: MicDiagnostics,
        HangingMugDiagnostics.task: HangingMugDiagnostics,
        ScanObjectDiagnostics.task: ScanObjectDiagnostics,
    }
    return adapters[task]()
