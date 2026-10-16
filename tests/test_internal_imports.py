"""Check that exported modules do not depend on an omitted project module."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"commvla": ROOT / "eagle/src", "pi05_fabric": ROOT / "pi05/src",
            "multiarm_sim": ROOT / "simulation/src"}


def test_all_project_imports_resolve_in_the_release():
    missing = []
    for base in PACKAGES.values():
        for path in base.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                    continue
                package = node.module.split(".")[0]
                if package not in PACKAGES:
                    continue
                target = PACKAGES[package].joinpath(*node.module.split("."))
                if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                    missing.append((str(path.relative_to(ROOT)), node.module))
    assert not missing
