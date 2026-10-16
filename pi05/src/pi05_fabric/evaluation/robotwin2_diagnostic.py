"""Pure helpers for fixed, no-retraining RoboTwin2 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from pi05_fabric.agents.dual_pi05 import DualPi05Mode
from pi05_fabric.evaluation.native_handover import native_inference_mode


@dataclass(frozen=True)
class DiagnosticSelection:
    condition_ids: tuple[int, ...]
    strata: Mapping[str, tuple[int, ...]]

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(values) for name, values in self.strata.items()}


def _spread(values: Iterable[int], count: int, *, name: str) -> tuple[int, ...]:
    ordered = tuple(sorted(set(values)))
    if len(ordered) < count:
        raise ValueError(
            f"diagnostic stratum {name} has {len(ordered)} conditions; {count} required"
        )
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=np.int64)
    return tuple(ordered[int(index)] for index in indices)


def select_diagnostic_condition_ids(
    rows: Iterable[Mapping[str, object]], *, per_stratum: int = 8
) -> DiagnosticSelection:
    """Select spread-out success and failure conditions from a completed evaluation."""
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    buckets = {
        "strict_success": [],
        "target_without_success": [],
        "pretarget_failure": [],
    }
    for row in rows:
        condition_id = int(row["condition_id"])
        success = bool(int(row["success"]))
        target = bool(int(row["target_region_reached"]))
        progressed = bool(int(row["block_lifted"])) or bool(int(row["middle_reached"]))
        if success:
            buckets["strict_success"].append(condition_id)
        elif target:
            buckets["target_without_success"].append(condition_id)
        elif progressed:
            buckets["pretarget_failure"].append(condition_id)
    strata = {
        name: _spread(values, per_stratum, name=name) for name, values in buckets.items()
    }
    condition_ids = tuple(sorted(value for values in strata.values() for value in values))
    return DiagnosticSelection(condition_ids=condition_ids, strata=strata)


def diagnostic_inference_mode(profile: str) -> DualPi05Mode:
    """Map a no-retraining profile name to the matching V2 inference mode."""
    return native_inference_mode(profile, v2=True)


def effective_interaction(mode: DualPi05Mode) -> dict[str, bool]:
    spec = mode.interaction_spec
    return {
        "common": bool(spec.common),
        "private_kv": bool(spec.private_kv),
        "peer_action_residual": bool(mode.remote_action_residual),
    }


def action_support_summary(
    actions_r6: np.ndarray, statistics: Mapping[str, object]
) -> dict[str, float | int]:
    """Count unnormalized actions outside the training q01/q99 support."""
    actions = np.asarray(actions_r6, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 20:
        raise ValueError("RoboTwin2 action support audit requires shape [H, 20]")
    q01 = np.asarray(
        statistics["left"]["action"]["q01"] + statistics["right"]["action"]["q01"],
        dtype=np.float32,
    )
    q99 = np.asarray(
        statistics["left"]["action"]["q99"] + statistics["right"]["action"]["q99"],
        dtype=np.float32,
    )
    below = int(np.count_nonzero(actions < q01[None, :]))
    above = int(np.count_nonzero(actions > q99[None, :]))
    values = int(actions.size)
    outside = below + above
    return {
        "values": values,
        "below_q01": below,
        "above_q99": above,
        "outside": outside,
        "outside_fraction": outside / values,
    }
