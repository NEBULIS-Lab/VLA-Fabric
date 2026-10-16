"""Launch a CommVLA torchrun job from a small YAML configuration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _expand(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)


def build_command(config: dict) -> tuple[list[str], dict[str, str]]:
    module = config["module"]
    processes = int(config.get("processes", 1))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={processes}",
        "-m",
        module,
    ]
    for key, value in config.get("arguments", {}).items():
        option = f"--{key.replace('_', '-')}"
        if value is True:
            command.append(option)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            command.extend([option, ",".join(map(str, value))])
        else:
            command.extend([option, str(value)])
    environment = os.environ.copy()
    environment.update({key: str(value) for key, value in config.get("environment", {}).items()})
    return command, environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _expand(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    unresolved = [
        value
        for value in _walk_strings(config)
        if "${" in value
    ]
    if unresolved:
        raise RuntimeError("Unset environment variables in configuration: " + ", ".join(sorted(set(unresolved))))
    command, environment = build_command(config)
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
