"""Direct RoboFactory HDF5 adapters for N-agent CommVLA training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_utils import BatchFeature


class RoboFactoryPaddedCollator:
    """Tensor-only inference/training collator without the TensorFlow RLDS import."""

    def __init__(self, pad_token_id: int = 1, padding_side: str = "right") -> None:
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> BatchFeature:
        if self.padding_side != "right":
            raise ValueError(f"Unsupported padding side: {self.padding_side}")
        output = BatchFeature({key: [] for key in instances[0]})
        for instance in instances:
            for key, value in instance.items():
                output[key].append(value)
        for key, values in output.items():
            if key == "dataset_name":
                continue
            if key in {"action", "pixel_values_primary", "pixel_values_wrist", "pixel_values"}:
                output[key] = torch.stack(values)
            elif key == "labels":
                output[key] = pad_sequence(values, batch_first=True, padding_value=-100)
            elif key == "input_ids":
                output[key] = pad_sequence(values, batch_first=True, padding_value=self.pad_token_id)
            else:
                output[key] = pad_sequence(values, batch_first=True, padding_value=0)
        return output


@dataclass
class RoboFactoryNAgentStatistics:
    proprio_low: np.ndarray
    proprio_high: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    @property
    def num_agents(self) -> int:
        return int(self.proprio_low.shape[0])

    @staticmethod
    def _normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return np.clip((values - low) * 2.0 / (high - low + 1e-6) - 1.0, -1.0, 1.0)

    def normalize_proprio(self, agent_idx: int, values: np.ndarray) -> np.ndarray:
        return self._normalize(values, self.proprio_low[agent_idx], self.proprio_high[agent_idx])

    def normalize_action(self, agent_idx: int, values: np.ndarray) -> np.ndarray:
        return self._normalize(values, self.action_low[agent_idx], self.action_high[agent_idx])

    @staticmethod
    def _unnormalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return (values + 1.0) * (high - low + 1e-6) / 2.0 + low

    def unnormalize_action(self, agent_idx: int, values: np.ndarray) -> np.ndarray:
        raw_dim = self.action_low.shape[-1]
        return self._unnormalize(values[..., :raw_dim], self.action_low[agent_idx], self.action_high[agent_idx])

    @staticmethod
    def infer_num_agents(h5_path: str | Path) -> int:
        with h5py.File(h5_path, "r") as handle:
            first = next(iter(handle.values()))
            indices = sorted(
                int(name.removeprefix("panda-"))
                for name in first["actions"].keys()
                if name.startswith("panda-")
            )
        if not indices or indices != list(range(len(indices))):
            raise ValueError(f"Expected contiguous panda agent IDs, got {indices}")
        return len(indices)

    @classmethod
    def compute(
        cls,
        h5_path: str | Path,
        num_agents: int | None = None,
    ) -> "RoboFactoryNAgentStatistics":
        if num_agents is None:
            num_agents = cls.infer_num_agents(h5_path)
        proprio = [[] for _ in range(num_agents)]
        action = [[] for _ in range(num_agents)]
        with h5py.File(h5_path, "r") as handle:
            for trajectory in handle.values():
                for agent_idx in range(num_agents):
                    proprio[agent_idx].append(trajectory[f"obs/agent/panda-{agent_idx}/qpos"][:])
                    action[agent_idx].append(trajectory[f"actions/panda-{agent_idx}"][:])
        proprio_array = [np.concatenate(items, axis=0) for items in proprio]
        action_array = [np.concatenate(items, axis=0) for items in action]
        return cls(
            proprio_low=np.stack([np.quantile(values, 0.01, axis=0) for values in proprio_array]),
            proprio_high=np.stack([np.quantile(values, 0.99, axis=0) for values in proprio_array]),
            action_low=np.stack([np.quantile(values, 0.01, axis=0) for values in action_array]),
            action_high=np.stack([np.quantile(values, 0.99, axis=0) for values in action_array]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(path) + ".tmp.npz")
        np.savez(
            temporary,
            proprio_low=self.proprio_low,
            proprio_high=self.proprio_high,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "RoboFactoryNAgentStatistics":
        with np.load(path) as values:
            return cls(
                proprio_low=values["proprio_low"],
                proprio_high=values["proprio_high"],
                action_low=values["action_low"],
                action_high=values["action_high"],
            )


@dataclass(frozen=True)
class RoboFactoryNAgentInputConfig:
    task_name: str
    global_instruction: str
    role_instructions: tuple[str, ...]
    global_camera: str
    local_cameras: tuple[str, ...]
    known_shared_local_views: tuple[tuple[int, ...], ...] = ()

    @property
    def num_agents(self) -> int:
        return len(self.role_instructions)

    @classmethod
    def load(cls, path: str | Path) -> "RoboFactoryNAgentInputConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agents = sorted(payload["agents"], key=lambda item: int(item["agent_id"]))
        agent_ids = [int(item["agent_id"]) for item in agents]
        if agent_ids != list(range(len(agents))):
            raise ValueError(f"Agent IDs must be contiguous from zero, got {agent_ids}")
        local_cameras = tuple(payload["cameras"]["agents"])
        role_instructions = tuple(item["role_instruction"] for item in agents)
        if len(local_cameras) != len(role_instructions):
            raise ValueError("The number of local cameras must equal the number of agent roles")
        declared_local = tuple(item.get("local_camera", local_cameras[idx]) for idx, item in enumerate(agents))
        if declared_local != local_cameras:
            raise ValueError("Agent-local camera declarations disagree with cameras.agents ordering")
        return cls(
            task_name=payload.get("task_name", "RoboFactory-rf"),
            global_instruction=payload["global_instruction"],
            role_instructions=role_instructions,
            global_camera=payload["cameras"]["global"],
            local_cameras=local_cameras,
            known_shared_local_views=tuple(
                tuple(int(agent_idx) for agent_idx in group)
                for group in payload["cameras"].get("known_shared_views", [])
            ),
        )


class RoboFactoryNAgentDataset:
    """Random-access action-chunk view over a RoboFactory N-agent HDF5 dataset."""

    def __init__(
        self,
        h5_path: str | Path,
        *,
        statistics: RoboFactoryNAgentStatistics,
        input_config: RoboFactoryNAgentInputConfig,
        action_len: int = 20,
        model_state_dim: int = 10,
        model_action_dim: int = 10,
    ) -> None:
        self.h5_path = str(h5_path)
        self.statistics = statistics
        self.input_config = input_config
        self.num_agents = input_config.num_agents
        if self.num_agents < 1:
            raise ValueError("At least one configured agent is required")
        if statistics.num_agents != self.num_agents:
            raise ValueError(
                f"Statistics contain {statistics.num_agents} agents, input config contains {self.num_agents}"
            )
        self.action_len = int(action_len)
        self.model_state_dim = int(model_state_dim)
        self.model_action_dim = int(model_action_dim)
        self._handle: h5py.File | None = None
        self._index: list[tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as handle:
            for trajectory_name, trajectory in handle.items():
                steps = int(trajectory["actions/panda-0"].shape[0])
                for agent_idx in range(self.num_agents):
                    action_steps = int(trajectory[f"actions/panda-{agent_idx}"].shape[0])
                    state_steps = int(trajectory[f"obs/agent/panda-{agent_idx}/qpos"].shape[0])
                    if action_steps != steps or state_steps < steps:
                        raise ValueError(
                            f"{trajectory_name} agent {agent_idx} has action/state lengths "
                            f"{action_steps}/{state_steps}, expected {steps}/at least {steps}"
                        )
                self._index.extend((trajectory_name, step) for step in range(steps))

    def __len__(self) -> int:
        return len(self._index)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except (TypeError, ValueError):
                pass
            self._handle = None

    def __del__(self):
        self.close()

    @staticmethod
    def _pad_last(values: np.ndarray, length: int) -> np.ndarray:
        if len(values) >= length:
            return values[:length]
        if len(values) == 0:
            raise ValueError("Cannot pad an empty action chunk")
        return np.concatenate([values, np.repeat(values[-1:], length - len(values), axis=0)], axis=0)

    @staticmethod
    def _pad_feature(values: np.ndarray, width: int) -> np.ndarray:
        if values.shape[-1] > width:
            raise ValueError(f"Cannot fit feature width {values.shape[-1]} into {width}")
        pad = width - values.shape[-1]
        return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)])

    def __getitem__(self, index: int) -> dict:
        trajectory_name, step = self._index[index]
        trajectory = self._get_handle()[trajectory_name]
        global_image = trajectory[f"obs/sensor_data/{self.input_config.global_camera}/rgb"][step]
        local_agent_images = []
        proprio = []
        actions = []
        for agent_idx in range(self.num_agents):
            camera_name = self.input_config.local_cameras[agent_idx]
            local_agent_images.append(trajectory[f"obs/sensor_data/{camera_name}/rgb"][step])
            state = trajectory[f"obs/agent/panda-{agent_idx}/qpos"][step : step + 1]
            state = self.statistics.normalize_proprio(agent_idx, state).astype(np.float32)
            proprio.append(self._pad_feature(state, self.model_state_dim))
            action = trajectory[f"actions/panda-{agent_idx}"][step : step + self.action_len]
            action = self._pad_last(action, self.action_len)
            action = self.statistics.normalize_action(agent_idx, action).astype(np.float32)
            actions.append(self._pad_feature(action, self.model_action_dim))
        return {
            "global_image": global_image[np.newaxis, :],
            "local_agent_images": [image[np.newaxis, :] for image in local_agent_images],
            "proprio": proprio,
            "actions": actions,
            "global_instruction": self.input_config.global_instruction,
            "role_instructions": self.input_config.role_instructions,
            "trajectory": trajectory_name,
            "step": step,
        }

    def make_agent_batches(self, model, indices: list[int]):
        return self.collate_samples(model, [self[index] for index in indices])

    def collate_samples(self, model, samples: list[dict]):
        if len(model.agents) != self.num_agents:
            raise ValueError(f"Model has {len(model.agents)} agents, dataset has {self.num_agents}")
        collator = RoboFactoryPaddedCollator()
        per_agent = [[] for _ in range(self.num_agents)]
        for sample in samples:
            processed = model.preprocess_inputs(
                sample["global_image"],
                sample["local_agent_images"],
                sample["global_instruction"],
                list(sample["role_instructions"]),
            )
            for agent_idx, inputs in enumerate(processed):
                instance = dict(inputs)
                instance["proprio"] = torch.from_numpy(sample["proprio"][agent_idx])
                instance["action"] = torch.from_numpy(sample["actions"][agent_idx])
                per_agent[agent_idx].append(instance)
        return [collator(instances) for instances in per_agent]


# Compatibility names for retained CameraAlignment scripts and checkpoints.
CameraAlignmentStatistics = RoboFactoryNAgentStatistics
CameraAlignmentInputConfig = RoboFactoryNAgentInputConfig
CameraAlignmentDataset = RoboFactoryNAgentDataset
