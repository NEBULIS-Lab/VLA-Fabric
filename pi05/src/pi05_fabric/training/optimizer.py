"""Optimizer and learning-rate schedule for formal pi0.5 LoRA runs."""

from __future__ import annotations

from dataclasses import dataclass

import optax
import jax.numpy as jnp

from pi05_fabric.agents.pi05_strong import ADAPTER_GROUP
from pi05_fabric.agents.pi05_strong import ACTION_EXPERT_GROUP
from pi05_fabric.agents.pi05_strong import ACTION_EXPERT_FFW_GROUP
from pi05_fabric.agents.pi05_strong import PALIGEMMA_GROUP
from pi05_fabric.agents.pi05_strong import PALIGEMMA_QO_GROUP


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
    initial_learning_rate: float | None = None
    schedule_offset: int = 0

    def validate(self) -> None:
        if not 0 <= self.schedule_offset < self.total_steps:
            raise ValueError('invalid schedule offset')
        if self.schedule_offset and self.warmup_steps:
            raise ValueError('offset extension cannot rewarm')
        if self.initial_learning_rate is not None and not 0 < self.initial_learning_rate <= self.peak_learning_rate:
            raise ValueError("initial_learning_rate must be positive and at most peak")
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
        decay = optax.cosine_decay_schedule(
            init_value=settings.peak_learning_rate,
            decay_steps=settings.total_steps - settings.schedule_offset,
            alpha=settings.final_learning_rate / settings.peak_learning_rate,
        )
        return lambda count: decay(jnp.maximum(0, count - settings.schedule_offset))
    return optax.warmup_cosine_decay_schedule(
        init_value=(settings.peak_learning_rate / (settings.warmup_steps + 1)
                    if settings.initial_learning_rate is None else settings.initial_learning_rate),
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
    action_ffw_peak: float = 5e-6
    action_ffw_final: float = 5e-7
    paligemma_peak: float = 3e-6
    paligemma_final: float = 3e-7
    paligemma_qo_peak: float = 3e-6
    paligemma_qo_final: float = 3e-7
    adapter_peak: float = 5e-5
    adapter_final: float = 5e-6
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 1e-5
    clip_gradient_norm: float = 1.0
    start_from_final: bool = False
    schedule_offset: int = 0

    def group_settings(self) -> dict[str, OptimizerSettings]:
        common = dict(
            schedule_offset=self.schedule_offset,
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
                initial_learning_rate=self.action_expert_final if self.start_from_final else None,
                **common,
            ),
            ACTION_EXPERT_FFW_GROUP: OptimizerSettings(
                peak_learning_rate=self.action_ffw_peak,
                final_learning_rate=self.action_ffw_final,
                initial_learning_rate=self.action_ffw_final if self.start_from_final else None,
                **common,
            ),
            PALIGEMMA_GROUP: OptimizerSettings(
                peak_learning_rate=self.paligemma_peak,
                final_learning_rate=self.paligemma_final,
                initial_learning_rate=self.paligemma_final if self.start_from_final else None,
                **common,
            ),
            PALIGEMMA_QO_GROUP: OptimizerSettings(
                peak_learning_rate=self.paligemma_qo_peak,
                final_learning_rate=self.paligemma_qo_final,
                initial_learning_rate=self.paligemma_qo_final if self.start_from_final else None,
                **common,
            ),
            ADAPTER_GROUP: OptimizerSettings(
                peak_learning_rate=self.adapter_peak,
                final_learning_rate=self.adapter_final,
                initial_learning_rate=self.adapter_final if self.start_from_final else None,
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
        # Optax <=0.2.4 exposes the same transform under its original name.
        partition = optax.multi_transform
    return optax.chain(
        optax.clip_by_global_norm(settings.clip_gradient_norm),
        partition(group_transforms, labels),
    )
