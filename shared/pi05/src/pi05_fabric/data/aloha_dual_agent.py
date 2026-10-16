"""ALOHA ownership adapter for two independent arm-side policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

COMMON = np.int8(0)
PRIVATE = np.int8(1)
LOCAL_DIM = 10
JOINT_DIM = 20
OPENPI_ACTION_DIM = 32


class _Pi05Tokenizer(Protocol):
    _max_len: int
    _tokenizer: object

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class AgentObservation:
    role: str
    images: Mapping[str, np.ndarray]
    proprioception: np.ndarray
    instruction: str


def _require_image(images: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in images:
        raise KeyError(f"missing required camera: {key}")
    return images[key]


def split_observation(sample: Mapping[str, object]) -> tuple[AgentObservation, AgentObservation]:
    """Split one bimanual sample without exposing peer-local observations."""

    images = sample["images"]
    if not isinstance(images, Mapping):
        raise TypeError("sample['images'] must be a mapping")
    global_image = _require_image(images, "global")
    left_wrist = _require_image(images, "left_wrist")
    right_wrist = _require_image(images, "right_wrist")

    proprioception = np.asarray(sample["proprioception"])
    if proprioception.shape[-1] != JOINT_DIM:
        raise ValueError(f"expected {JOINT_DIM}D joint proprioception, got {proprioception.shape[-1]}")
    instruction = str(sample["instruction"])

    left = AgentObservation(
        role="left",
        images={"global": global_image, "wrist": left_wrist},
        proprioception=proprioception[..., :LOCAL_DIM],
        instruction=instruction,
    )
    right = AgentObservation(
        role="right",
        images={"global": global_image, "wrist": right_wrist},
        proprioception=proprioception[..., LOCAL_DIM:],
        instruction=instruction,
    )
    return left, right


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action)
    if action.shape[-1] != JOINT_DIM:
        raise ValueError(f"expected {JOINT_DIM}D joint action, got {action.shape[-1]}")
    return action[..., :LOCAL_DIM], action[..., LOCAL_DIM:]


def merge_action(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.shape[-1] != LOCAL_DIM:
        raise ValueError("left and right actions must have equal shape ending in 10")
    return np.concatenate([left, right], axis=-1)


def pad_local_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action)
    if action.shape[-1] != LOCAL_DIM:
        raise ValueError(f"expected {LOCAL_DIM}D local action, got {action.shape[-1]}")
    padding = [(0, 0)] * action.ndim
    padding[-1] = (0, OPENPI_ACTION_DIM - LOCAL_DIM)
    return np.pad(action, padding, mode="constant")


def expose_local_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action)
    if action.shape[-1] != OPENPI_ACTION_DIM:
        raise ValueError(f"expected {OPENPI_ACTION_DIM}D internal action, got {action.shape[-1]}")
    return action[..., :LOCAL_DIM]


def build_prefix_ownership(
    *,
    global_image_tokens: int,
    wrist_image_tokens: int,
    instruction_tokens: int,
    state_tokens: int,
) -> np.ndarray:
    """Mark global-image/instruction tokens common and wrist/state tokens private."""

    lengths = (global_image_tokens, wrist_image_tokens, instruction_tokens, state_tokens)
    if any(not isinstance(length, int) or length < 0 for length in lengths):
        raise ValueError("token counts must be non-negative integers")
    return np.concatenate(
        [
            np.full(global_image_tokens, COMMON, dtype=np.int8),
            np.full(wrist_image_tokens, PRIVATE, dtype=np.int8),
            np.full(instruction_tokens, COMMON, dtype=np.int8),
            np.full(state_tokens, PRIVATE, dtype=np.int8),
        ]
    )


def tokenize_pi05_prompt_with_ownership(
    tokenizer: _Pi05Tokenizer,
    prompt: str,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize exactly as OpenPI and classify the state-bearing suffix Private."""

    tokens, token_mask = tokenizer.tokenize(prompt, state)
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    common_prefix = f"Task: {cleaned}, State:"
    prefix_tokens = tokenizer._tokenizer.encode(common_prefix, add_bos=True)
    active = np.asarray(tokens)[np.asarray(token_mask, dtype=bool)]
    common_count = 0
    for actual, expected in zip(active, prefix_tokens, strict=False):
        if int(actual) != int(expected):
            break
        common_count += 1

    ownership = np.full(np.asarray(tokens).shape, PRIVATE, dtype=np.int8)
    ownership[:common_count] = COMMON
    return np.asarray(tokens), np.asarray(token_mask), ownership


def build_openpi_prefix_ownership(
    *,
    image_names: tuple[str, ...],
    image_tokens_per_view: int | Mapping[str, int],
    text_ownership: np.ndarray,
) -> np.ndarray:
    """Prepend OpenPI image-token ownership in the model's dictionary order."""

    text = np.asarray(text_ownership, dtype=np.int8)
    if text.ndim not in (1, 2):
        raise ValueError("text ownership must have shape [tokens] or [batch, tokens]")
    batch = 1 if text.ndim == 1 else text.shape[0]
    image_parts = []
    for name in image_names:
        count = image_tokens_per_view[name] if isinstance(image_tokens_per_view, Mapping) else image_tokens_per_view
        if not isinstance(count, int) or count < 0:
            raise ValueError("image token counts must be non-negative integers")
        value = COMMON if name == "base_0_rgb" else PRIVATE
        image_parts.append(np.full((batch, count), value, dtype=np.int8))
    text_batch = text[None, :] if text.ndim == 1 else text
    combined = np.concatenate([*image_parts, text_batch], axis=1)
    return combined[0] if text.ndim == 1 else combined
