"""Paired summaries for Full, Core, and Common-only evaluations."""

from __future__ import annotations

from typing import Iterable, Mapping

from pi05_fabric.evaluation.protocol import audit_rollout_rows


PROFILE_ORDER = ("full", "core", "common_only")
COMPARISONS = (
    ("full", "core"),
    ("core", "common_only"),
    ("full", "common_only"),
)


def _profile_summary(rows: tuple[Mapping[str, object], ...]) -> dict[str, object]:
    successes = sum(bool(int(row["success"])) for row in rows)
    trials = len(rows)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials,
        "timeouts": sum(int(row["timeout"]) for row in rows),
        "errors": sum(bool(str(row["error"])) for row in rows),
    }


def compare_reference_outcomes(
    current: Iterable[Mapping[str, object]],
    reference: Iterable[Mapping[str, object]],
    *,
    expected_ids: tuple[int, ...],
) -> dict[str, object]:
    current_rows = tuple(current)
    reference_rows = tuple(reference)
    audit_rollout_rows(current_rows, expected_ids=expected_ids)
    audit_rollout_rows(reference_rows, expected_ids=expected_ids)
    current_success = {
        int(row["condition_id"]) for row in current_rows if bool(int(row["success"]))
    }
    reference_success = {
        int(row["condition_id"]) for row in reference_rows if bool(int(row["success"]))
    }
    changed = sorted(current_success ^ reference_success)
    trials = len(expected_ids)
    return {
        "exact_match": not changed,
        "current_successes": len(current_success),
        "reference_successes": len(reference_success),
        "delta_percentage_points": round(
            100.0 * (len(current_success) - len(reference_success)) / trials, 6
        ),
        "changed_conditions": changed,
        "current_only_success": sorted(current_success - reference_success),
        "reference_only_success": sorted(reference_success - current_success),
    }


def summarize_channel_ablation(
    profiles: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    expected_ids: tuple[int, ...],
) -> dict[str, object]:
    if set(profiles) != set(PROFILE_ORDER):
        raise ValueError(f"profiles must be exactly: {', '.join(PROFILE_ORDER)}")

    indexed: dict[str, dict[int, Mapping[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for profile in PROFILE_ORDER:
        rows = tuple(profiles[profile])
        audit_rollout_rows(rows, expected_ids=expected_ids)
        if any(not bool(int(row["finite_all"])) for row in rows):
            raise ValueError(f"{profile} contains non-finite rollout output")
        indexed[profile] = {int(row["condition_id"]): row for row in rows}
        summaries[profile] = _profile_summary(rows)

    comparisons: dict[str, dict[str, object]] = {}
    for left, right in COMPARISONS:
        left_success = {
            condition_id
            for condition_id, row in indexed[left].items()
            if bool(int(row["success"]))
        }
        right_success = {
            condition_id
            for condition_id, row in indexed[right].items()
            if bool(int(row["success"]))
        }
        comparisons[f"{left}_vs_{right}"] = {
            "left": left,
            "right": right,
            "delta_percentage_points": round(
                100.0
                * (summaries[left]["success_rate"] - summaries[right]["success_rate"]),
                6,
            ),
            "both_success": len(left_success & right_success),
            "left_only": len(left_success - right_success),
            "right_only": len(right_success - left_success),
            "both_fail": len(set(expected_ids) - left_success - right_success),
        }

    return {
        "condition_ids": list(expected_ids),
        "profiles": summaries,
        "comparisons": comparisons,
    }
