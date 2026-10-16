from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
DIRECT = {
    "I0",
    "I1",
    "I2",
    "I3-AVG",
    "I3-RAW",
    "I4-AVG",
    "I4-RAW",
    "I5-AVG",
    "I5-RAW",
}
COMMON_FIRST = {
    "I2",
    "I3-AVG",
    "I3-RAW",
    "I4-AVG",
    "I4-RAW",
    "I5-AVG",
    "I5-RAW",
}


def _matrix(task: str) -> dict:
    path = ROOT / "configs" / "matrix" / f"{task}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_both_tasks_cover_the_reported_sixteen_model_matrix() -> None:
    for task in ("aloha_handover_box", "aloha_shoes_table"):
        matrix = _matrix(task)
        assert set(matrix["single_stage"]) == DIRECT
        assert set(matrix["common_first"]) == COMMON_FIRST


def test_selected_long_run_scope_is_i4_and_i5_raw_only() -> None:
    configs = {
        path.stem for path in (ROOT / "configs" / "runs" / "unified_100k").glob("*.yaml")
    }
    assert configs == {
        "handover_i4_raw",
        "handover_i5_raw",
        "shoes_i4_raw",
        "shoes_i5_raw",
    }
