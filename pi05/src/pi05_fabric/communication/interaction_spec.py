"""Canonical four-path interaction contract for paired VLA agents."""

from __future__ import annotations

from typing import NamedTuple


class InteractionSpec(NamedTuple):
    common: bool
    private_kv: bool
    remote_action_kv: bool
    final_linear_action: bool
