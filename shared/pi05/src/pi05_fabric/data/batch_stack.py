"""Batch collation for paired pi0.5 samples."""

from __future__ import annotations

import numpy as np

from openpi.models import model as model_api

from pi05_fabric.data.pi05_batch import Pi05TrainingPair


def _stack_observations(observations):
    return model_api.Observation(
        images={
            key: np.concatenate([item.images[key] for item in observations], axis=0)
            for key in observations[0].images
        },
        image_masks={
            key: np.concatenate([item.image_masks[key] for item in observations], axis=0)
            for key in observations[0].image_masks
        },
        state=np.concatenate([item.state for item in observations], axis=0),
        tokenized_prompt=np.concatenate([item.tokenized_prompt for item in observations], axis=0),
        tokenized_prompt_mask=np.concatenate([item.tokenized_prompt_mask for item in observations], axis=0),
    )


def stack_training_pairs(pairs: list[Pi05TrainingPair]) -> Pi05TrainingPair:
    if not pairs:
        raise ValueError("at least one paired sample is required")
    return Pi05TrainingPair(
        _stack_observations([pair.left_observation for pair in pairs]),
        _stack_observations([pair.right_observation for pair in pairs]),
        np.concatenate([pair.left_actions for pair in pairs], axis=0),
        np.concatenate([pair.right_actions for pair in pairs], axis=0),
        np.concatenate([pair.ownership for pair in pairs], axis=0),
    )
