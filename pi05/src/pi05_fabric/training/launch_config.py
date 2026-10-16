"""Validated launch configuration for dual pi0.5 training stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.training.protocol import checkpoint_steps
from pi05_fabric.training.stages import StageName


_MODES = {
    StageName.INDEPENDENT: DualPi05Mode.INDEPENDENT,
    StageName.RAW_COMMON_STAGE1: DualPi05Mode.RAW_COMMON_ONLY,
    StageName.I3_RAW_CORE_STAGE2: DualPi05Mode.I3_RAW_CORE,
    StageName.I4_RAW_FULL_STAGE2: DualPi05Mode.I4_RAW_FULL,
    StageName.PI_NATIVE_STRONG_INDEPENDENT: DualPi05Mode.PI_NATIVE_INDEPENDENT,
    StageName.PI_NATIVE_V2_INDEPENDENT: DualPi05Mode.PI_NATIVE_V2_INDEPENDENT,
    StageName.PI_NATIVE_RAW_COMMON_STAGE1: DualPi05Mode.PI_NATIVE_RAW_COMMON_ONLY,
    StageName.PI_NATIVE_THREE_PATH_STAGE2: DualPi05Mode.PI_NATIVE_THREE_PATH,
    StageName.PI_NATIVE_V2_RAW_COMMON_STAGE1: DualPi05Mode.PI_NATIVE_V2_RAW_COMMON_ONLY,
    StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2: DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
    StageName.PI_NATIVE_V2_FULL_DIRECT: DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
    StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION: DualPi05Mode.PI_NATIVE_V2_RESIDUAL_ACTION,
}


@dataclass(frozen=True)
class LaunchConfig:
    stage: StageName
    dataset: Path
    base_checkpoint: Path
    output: Path
    steps: int
    batch_size: int
    learning_rate: float
    seed: int | None = None
    model_seed: int | None = None
    training_seed: int | None = None
    parent_training_seed: int | None = None
    final_learning_rate: float | None = None
    warmup_steps: int = 0
    device_count: int = 1
    gradient_accumulation: int = 1
    checkpoint_every: int = 0
    stage1_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    weights_checkpoint: Path | None = None

    @property
    def mode(self) -> DualPi05Mode:
        return _MODES[self.stage]

    @property
    def global_batch_size(self) -> int:
        return self.batch_size * self.device_count * self.gradient_accumulation

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        if self.checkpoint_every <= 0:
            return (self.steps,)
        return checkpoint_steps(total_steps=self.steps, every=self.checkpoint_every)

    @property
    def resolved_model_seed(self) -> int:
        if self.model_seed is not None:
            return self.model_seed
        if self.seed is not None:
            return self.seed
        raise ValueError("model_seed is required")

    @property
    def resolved_training_seed(self) -> int:
        if self.training_seed is not None:
            return self.training_seed
        if self.seed is not None:
            return self.seed
        raise ValueError("training_seed is required")

    def validate_paths(self, *, require_existing: bool = True) -> None:
        self.resolved_model_seed
        self.resolved_training_seed
        if self.steps <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("steps, batch_size, and learning_rate must be positive")
        if self.device_count <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("device_count and gradient_accumulation must be positive")
        final_lr = self.learning_rate if self.final_learning_rate is None else self.final_learning_rate
        if final_lr <= 0 or final_lr > self.learning_rate:
            raise ValueError("final_learning_rate must be positive and no larger than learning_rate")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every cannot be negative")
        if self.stage is StageName.PI_NATIVE_V2_FULL_DIRECT:
            if self.stage1_checkpoint is not None or self.weights_checkpoint is not None:
                raise ValueError("Full-direct requires base initialization or same-stage resume, not stage1/weights")
            if self.parent_training_seed is not None:
                raise ValueError("Full-direct has no parent training seed")
        if self.stage is StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION:
            if self.weights_checkpoint is None and self.resume_checkpoint is None:
                raise ValueError("expanded continuation requires a weight parent or same-stage resume")
            if self.stage1_checkpoint is not None:
                raise ValueError("expanded continuation requires a weight parent, not a Stage 1 fork")
        if self.stage in (
            StageName.I3_RAW_CORE_STAGE2,
            StageName.I4_RAW_FULL_STAGE2,
            StageName.PI_NATIVE_THREE_PATH_STAGE2,
            StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        ):
            if self.stage1_checkpoint is None and self.resume_checkpoint is None and self.weights_checkpoint is None:
                raise ValueError("stage1_checkpoint is required for a new Stage 2 run")
        if (
            self.stage in (
                StageName.PI_NATIVE_THREE_PATH_STAGE2,
                StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
            )
            and self.parent_training_seed is not None
            and self.resolved_training_seed == self.parent_training_seed
        ):
            raise ValueError("Stage 2 training seed must differ from the Stage 1 training seed")
        if sum(p is not None for p in (self.stage1_checkpoint, self.resume_checkpoint, self.weights_checkpoint)) > 1:
            raise ValueError("Stage 2 fork and same-stage resume are mutually exclusive")
        if require_existing:
            for name, path in (("dataset", self.dataset), ("base_checkpoint", self.base_checkpoint)):
                if not path.exists():
                    raise FileNotFoundError(f"{name} does not exist: {path}")
            for name, path in (("stage1_checkpoint", self.stage1_checkpoint), ("resume_checkpoint", self.resume_checkpoint), ("weights_checkpoint", self.weights_checkpoint)):
                if path is not None and not path.exists():
                    raise FileNotFoundError(f"{name} does not exist: {path}")
        if self.output.exists() and self.resume_checkpoint is None:
            raise FileExistsError(f"output already exists: {self.output}")
