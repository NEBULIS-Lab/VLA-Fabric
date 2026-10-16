"""Verification helpers for linked experiment dependencies."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_external_sources(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> list[str]:
    """Return dependency errors without mutating linked sources."""

    manifest_path = Path(manifest_path)
    root = Path(project_root) if project_root is not None else manifest_path.parent.parent
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    sources = manifest.get("sources", {})
    errors: list[str] = []

    for source_name, source in sorted(sources.items()):
        link = root / source["link"]
        if not link.is_symlink():
            errors.append(f"{source_name}: expected symbolic link at {link}")
            continue
        if not link.exists():
            errors.append(f"{source_name}: broken symbolic link at {link}")
            continue

        for relative_path, expected in sorted(source.get("key_files", {}).items()):
            file_path = link / relative_path
            if not file_path.is_file():
                errors.append(f"{source_name}: missing key file {relative_path}")
                continue
            observed = _sha256(file_path)
            if observed != expected:
                errors.append(
                    f"{source_name}: SHA256 mismatch for {relative_path}: "
                    f"expected {expected}, observed {observed}"
                )

    return errors
