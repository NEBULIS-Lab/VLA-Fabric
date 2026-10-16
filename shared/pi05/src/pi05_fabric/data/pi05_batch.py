"""Build privacy-preserving OpenPI batches from paired local samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.models import model as model_api

from pi05_fabric.data.aloha_dual_agent import build_openpi_prefix_ownership
from pi05_fabric.data.aloha_dual_agent import pad_local_action
from pi05_fabric.data.aloha_dual_agent import tokenize_pi05_prompt_with_ownership
from pi05_fabric.data.converted_dataset import PairedTrainingSample


@dataclass(frozen=True)
class _Stats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "_Stats":
        mean = np.asarray(value["mean"], dtype=np.float32)
        std = np.asarray(value["std"], dtype=np.float32)
        if mean.shape != (10,) or std.shape != (10,):
            raise ValueError("local normalization statistics must be 10D")
        if np.any(std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        return cls(mean, std)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) - self.mean) / self.std

    def unnormalize(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * self.std + self.mean


@dataclass(frozen=True)
class _QuantileStats:
    q01: np.ndarray
    q99: np.ndarray
    epsilon: float

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, epsilon: float) -> "_QuantileStats":
        q01 = np.asarray(value["q01"], dtype=np.float32)
        q99 = np.asarray(value["q99"], dtype=np.float32)
        if q01.shape != (10,) or q99.shape != (10,):
            raise ValueError("local q01/q99 statistics must be 10D")
        if not np.isfinite(q01).all() or not np.isfinite(q99).all():
            raise ValueError("q01/q99 statistics must be finite")
        if np.any(q99 <= q01):
            raise ValueError("every q99 value must exceed q01")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("quantile normalization epsilon must be positive")
        return cls(q01, q99, float(epsilon))

    def normalize(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return (value - self.q01) / (self.q99 - self.q01 + self.epsilon) * 2.0 - 1.0

    def unnormalize(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return (value + 1.0) / 2.0 * (self.q99 - self.q01 + self.epsilon) + self.q01


@dataclass(frozen=True)
class LocalQuantileNormalization:
    left_state: _QuantileStats
    left_action: _QuantileStats
    right_state: _QuantileStats
    right_action: _QuantileStats
    dataset_manifest_sha256: str

    @classmethod
    def from_json(cls, path: str | Path) -> "LocalQuantileNormalization":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 2 or data.get("normalization") != "q01_q99":
            raise ValueError("unsupported q01/q99 normalization schema")
        epsilon = float(data.get("epsilon", 1e-6))
        manifest_sha = str(data.get("dataset_manifest_sha256", ""))
        if not manifest_sha:
            raise ValueError("quantile statistics require a dataset manifest SHA256")
        return cls(
            _QuantileStats.from_dict(data["left"]["state"], epsilon=epsilon),
            _QuantileStats.from_dict(data["left"]["action"], epsilon=epsilon),
            _QuantileStats.from_dict(data["right"]["state"], epsilon=epsilon),
            _QuantileStats.from_dict(data["right"]["action"], epsilon=epsilon),
            manifest_sha,
        )

    def unnormalize_actions(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.left_action.unnormalize(np.asarray(left)[..., :10]),
            self.right_action.unnormalize(np.asarray(right)[..., :10]),
        )


@dataclass(frozen=True)
class LocalNormalization:
    left_state: _Stats
    left_action: _Stats
    right_state: _Stats
    right_action: _Stats

    @classmethod
    def from_json(cls, path: str | Path) -> "LocalNormalization":
        data = json.loads(Path(path).read_text())
        return cls(
            _Stats.from_dict(data["left"]["state"]),
            _Stats.from_dict(data["left"]["action"]),
            _Stats.from_dict(data["right"]["state"]),
            _Stats.from_dict(data["right"]["action"]),
        )

    def unnormalize_actions(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.left_action.unnormalize(np.asarray(left)[..., :10]),
            self.right_action.unnormalize(np.asarray(right)[..., :10]),
        )


@dataclass(frozen=True)
class Pi05TrainingPair:
    left_observation: model_api.Observation[np.ndarray]
    right_observation: model_api.Observation[np.ndarray]
    left_actions: np.ndarray
    right_actions: np.ndarray
    ownership: np.ndarray


def _pad_state(state: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(state, dtype=np.float32), (0, 22))


def _local_observation(agent, state: np.ndarray, tokens: np.ndarray, token_mask: np.ndarray):
    black = np.zeros_like(agent.images["wrist"])
    images = {
        "base_0_rgb": agent.images["global"][None, ...],
        "left_wrist_0_rgb": (agent.images["wrist"] if agent.role == "left" else black)[None, ...],
        "right_wrist_0_rgb": (agent.images["wrist"] if agent.role == "right" else black)[None, ...],
    }
    masks = {
        "base_0_rgb": np.asarray([True]),
        "left_wrist_0_rgb": np.asarray([agent.role == "left"]),
        "right_wrist_0_rgb": np.asarray([agent.role == "right"]),
    }
    return model_api.Observation.from_dict(
        {
            "image": images,
            "image_mask": masks,
            "state": _pad_state(state)[None, ...],
            "tokenized_prompt": tokens[None, ...].astype(np.int32),
            "tokenized_prompt_mask": token_mask[None, ...].astype(bool),
        }
    )


def build_training_pair(
    sample: PairedTrainingSample,
    *,
    normalization: LocalNormalization | LocalQuantileNormalization,
    tokenizer,
    image_tokens_per_view: int = 256,
) -> Pi05TrainingPair:
    """Normalize, tokenize, and batch one paired sample for two complete policies."""

    left_state = normalization.left_state.normalize(sample.left.proprioception)
    right_state = normalization.right_state.normalize(sample.right.proprioception)
    left_tokens, left_mask, text_ownership = tokenize_pi05_prompt_with_ownership(
        tokenizer, sample.left.instruction, left_state
    )
    right_tokens, right_mask, right_ownership = tokenize_pi05_prompt_with_ownership(
        tokenizer, sample.right.instruction, right_state
    )
    if not np.array_equal(text_ownership, right_ownership):
        raise ValueError("left and right prompt ownership layouts differ")
    ownership = build_openpi_prefix_ownership(
        image_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        image_tokens_per_view=image_tokens_per_view,
        text_ownership=text_ownership,
    )[None, ...]
    left_observation = _local_observation(sample.left, left_state, left_tokens, left_mask)
    right_observation = _local_observation(sample.right, right_state, right_tokens, right_mask)
    left_actions = pad_local_action(normalization.left_action.normalize(sample.left_action))[None, ...]
    right_actions = pad_local_action(normalization.right_action.normalize(sample.right_action))[None, ...]
    return Pi05TrainingPair(
        left_observation,
        right_observation,
        left_actions.astype(np.float32),
        right_actions.astype(np.float32),
        ownership,
    )
