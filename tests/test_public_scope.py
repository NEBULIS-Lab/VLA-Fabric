"""Keep robot-specific implementation out of the public code release."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MODULES = {
    "multi_arm_episode", "export_multi_arm_pi05",
    "pi05_sampling", "validate_multi_arm_episode",
}


def test_no_physical_robot_code_is_included():
    assert not any(path.is_file() for path in (ROOT / "physical").rglob("*"))
    for directory in ("eagle", "pi05", "shared", "simulation"):
        assert not any(path.stem in PRIVATE_MODULES for path in (ROOT / directory).rglob("*.py"))


def test_public_code_does_not_import_private_robot_modules():
    for directory in ("eagle", "pi05", "shared", "simulation"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                assert not any(set(name.split(".")) & PRIVATE_MODULES for name in names), path
