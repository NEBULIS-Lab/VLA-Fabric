"""Optimizer and learning-rate schedule for formal pi0.5 LoRA runs."""

from __future__ import annotations

from dataclasses import dataclass

import optax

from pi05_fabric.agents.pi05_strong import ADAPTER_GROUP
from pi05_fabric.agents.pi05_strong import ACTION_EXPERT_GROUP
from pi05_fabric.agents.pi05_strong import PALIGEMMA_GROUP


@dataclass(frozen=True)
class OptimizerSettings:
    peak_learning_rate: float
    final_learning_rate: float
    warmup_steps: int
    total_steps: int
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 1e-5
    clip_gradient_norm: float = 1.0

    def validate(self) -> None:
        if self.peak_learning_rate <= 0 or self.final_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.final_learning_rate > self.peak_learning_rate:
            raise ValueError("final learning rate cannot exceed peak learning rate")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")


def create_learning_rate_schedule(settings: OptimizerSettings) -> optax.Schedule:
    settings.validate()
    if settings.warmup_steps == 0:
        return optax.cosine_decay_schedule(
            init_value=settings.peak_learning_rate,
            decay_steps=settings.total_steps,
            alpha=settings.final_learning_rate / settings.peak_learning_rate,
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=settings.peak_learning_rate / (settings.warmup_steps + 1),
        peak_value=settings.peak_learning_rate,
        warmup_steps=settings.warmup_steps,
        decay_steps=settings.total_steps,
        end_value=settings.final_learning_rate,
    )


def create_optimizer(settings: OptimizerSettings) -> optax.GradientTransformation:
    schedule = create_learning_rate_schedule(settings)
    adamw = optax.adamw(
        schedule,
        b1=settings.beta1,
        b2=settings.beta2,
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
    )
    return optax.chain(optax.clip_by_global_norm(settings.clip_gradient_norm), adamw)


@dataclass(frozen=True)
class GroupedOptimizerSettings:
    total_steps: int
    warmup_steps: int
    action_expert_peak: float = 5e-6
    action_expert_final: float = 5e-7
    paligemma_peak: float = 3e-6
    paligemma_final: float = 3e-7
    adapter_peak: float = 5e-5
    adapter_final: float = 5e-6
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 1e-5
    clip_gradient_norm: float = 1.0

    def group_settings(self) -> dict[str, OptimizerSettings]:
        common = dict(
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            clip_gradient_norm=self.clip_gradient_norm,
        )
        return {
            ACTION_EXPERT_GROUP: OptimizerSettings(
                peak_learning_rate=self.action_expert_peak,
                final_learning_rate=self.action_expert_final,
                **common,
            ),
            PALIGEMMA_GROUP: OptimizerSettings(
                peak_learning_rate=self.paligemma_peak,
                final_learning_rate=self.paligemma_final,
                **common,
            ),
            ADAPTER_GROUP: OptimizerSettings(
                peak_learning_rate=self.adapter_peak,
                final_learning_rate=self.adapter_final,
                **common,
            ),
        }


def create_grouped_optimizer(
    settings: GroupedOptimizerSettings,
    labels,
) -> optax.GradientTransformation:
    """Clip the complete gradient once, then apply disjoint group schedules."""
    group_transforms = {}
    for name, group in settings.group_settings().items():
        schedule = create_learning_rate_schedule(group)
        group_transforms[name] = optax.adamw(
            schedule,
            b1=group.beta1,
            b2=group.beta2,
            eps=group.epsilon,
            weight_decay=group.weight_decay,
        )
    partition = getattr(optax, "partition", None)
    if partition is None:
        # Optax <= 0.2.4 exposes this transformation under its original name.
        partition = optax.multi_transform
    return optax.chain(
        optax.clip_by_global_norm(settings.clip_gradient_norm),
        partition(group_transforms, labels),
    )
