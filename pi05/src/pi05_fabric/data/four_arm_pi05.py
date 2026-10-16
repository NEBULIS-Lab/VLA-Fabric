"""Build role-local OpenPI batches for four-arm MuJoCo tasks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from openpi.models import model as model_api

from pi05_fabric.data.aloha_dual_agent import build_openpi_prefix_ownership
from pi05_fabric.data.aloha_dual_agent import tokenize_pi05_prompt_with_ownership
from pi05_fabric.data.four_arm_tasks import FourArmQuantileNormalization
from pi05_fabric.data.four_arm_tasks import normalize_quantile


OPENPI_DIM = 32




@dataclass(frozen=True)
class Pi05RoleTrainingSample:
    observation: model_api.Observation[np.ndarray]
    actions: np.ndarray
    ownership: np.ndarray
    role_index: int


def _pad_last(value, *, target: int):
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] > target:
        raise ValueError("local vector is wider than the PI0.5 boundary")
    padding = [(0, 0)] * value.ndim
    padding[-1] = (0, target - value.shape[-1])
    return np.pad(value, padding, mode="constant")


def build_role_policy_observation(
    *,
    role_index: int,
    global_rgb,
    wrist_rgb,
    state,
    normalization: FourArmQuantileNormalization,
    tokenizer,
    instruction: str,
    image_tokens_per_view: int = 256,
):
    """Build the identical role-local PI0.5 boundary for train and rollout."""

    role_stats = normalization.for_role(role_index)
    normalized_state = normalize_quantile(state, role_stats.state)
    tokens, token_mask, text_ownership = tokenize_pi05_prompt_with_ownership(
        tokenizer,
        instruction,
        normalized_state,
    )
    ownership = build_openpi_prefix_ownership(
        image_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        image_tokens_per_view=image_tokens_per_view,
        text_ownership=text_ownership,
    )[None]
    global_rgb = np.asarray(global_rgb)
    wrist_rgb = np.asarray(wrist_rgb)
    if global_rgb.ndim != 3 or wrist_rgb.ndim != 3:
        raise ValueError("role-local RGB inputs must have shape [height, width, channels]")
    black = np.zeros_like(wrist_rgb)
    observation = model_api.Observation.from_dict(
        {
            "image": {
                "base_0_rgb": global_rgb[None],
                "left_wrist_0_rgb": wrist_rgb[None],
                "right_wrist_0_rgb": black[None],
            },
            "image_mask": {
                "base_0_rgb": np.asarray([True]),
                "left_wrist_0_rgb": np.asarray([True]),
                "right_wrist_0_rgb": np.asarray([False]),
            },
            "state": _pad_last(normalized_state, target=OPENPI_DIM)[None],
            "tokenized_prompt": tokens[None].astype(np.int32),
            "tokenized_prompt_mask": token_mask[None].astype(bool),
        }
    )
    return observation, ownership


def build_role_training_sample(
    sample,
    *,
    normalization: FourArmQuantileNormalization,
    tokenizer,
    instruction: str,
    image_tokens_per_view: int = 256,
) -> Pi05RoleTrainingSample:
    role_index = int(sample.role_index)
    role_stats = normalization.for_role(role_index)
    observation, ownership = build_role_policy_observation(
        role_index=role_index,
        global_rgb=sample.global_rgb,
        wrist_rgb=sample.wrist_rgb,
        state=sample.state,
        normalization=normalization,
        tokenizer=tokenizer,
        instruction=instruction,
        image_tokens_per_view=image_tokens_per_view,
    )
    actions = normalize_quantile(sample.actions, role_stats.action)
    actions = _pad_last(actions, target=OPENPI_DIM)[None]
    return Pi05RoleTrainingSample(
        observation=observation,
        actions=actions.astype(np.float32),
        ownership=ownership,
        role_index=role_index,
    )


def _stack_observations(observations):
    return model_api.Observation(
        images={
            key: np.concatenate([item.images[key] for item in observations], axis=0)
            for key in observations[0].images
        },
        image_masks={
            key: np.concatenate(
                [item.image_masks[key] for item in observations], axis=0
            )
            for key in observations[0].image_masks
        },
        state=np.concatenate([item.state for item in observations], axis=0),
        tokenized_prompt=np.concatenate(
            [item.tokenized_prompt for item in observations], axis=0
        ),
        tokenized_prompt_mask=np.concatenate(
            [item.tokenized_prompt_mask for item in observations], axis=0
        ),
    )


def stack_role_training_samples(
    samples: list[Pi05RoleTrainingSample],
) -> Pi05RoleTrainingSample:
    if not samples:
        raise ValueError("at least one role-local sample is required")
    roles = {sample.role_index for sample in samples}
    if len(roles) != 1:
        raise ValueError("one rank may batch samples for only one role")
    return Pi05RoleTrainingSample(
        observation=_stack_observations([sample.observation for sample in samples]),
        actions=np.concatenate([sample.actions for sample in samples], axis=0),
        ownership=np.concatenate([sample.ownership for sample in samples], axis=0),
        role_index=samples[0].role_index,
    )
