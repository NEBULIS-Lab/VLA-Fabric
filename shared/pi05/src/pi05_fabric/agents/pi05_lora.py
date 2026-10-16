"""Project-owned trainable parameter boundary for dual-agent pi0.5."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import nnx


_LOCAL_ACTION_MODULES = frozenset(
    {
        "action_in_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_out_proj",
    }
)


def _project_path_filter(*, include_linear_action: bool) -> Callable[[tuple[Any, ...], Any], bool]:
    def matches(path: tuple[Any, ...], _value: Any) -> bool:
        segments = tuple(str(part) for part in path)
        if any("lora" in segment.lower() for segment in segments):
            return True
        if any(segment in _LOCAL_ACTION_MODULES for segment in segments):
            return True
        return include_linear_action and any(
            segment.startswith("fabric_linear_action") for segment in segments
        )

    return matches


def project_trainable_filter(*, include_linear_action: bool = False) -> nnx.filterlib.Filter:
    """Select LoRA and receiver-local action modules while freezing both bases."""

    return nnx.All(
        nnx.Param,
        _project_path_filter(include_linear_action=include_linear_action),
    )
