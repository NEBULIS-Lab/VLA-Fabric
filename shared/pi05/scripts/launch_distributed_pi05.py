#!/usr/bin/env python3
"""Launch one complete dual-agent PI0.5 replica per GPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU indices")
    parser.add_argument("--coordinator", default="127.0.0.1:29605")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain at least two distinct GPU indices")
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args.pop(0)
    if not trainer_args:
        raise ValueError("trainer arguments are required after --")

    project_root = Path(__file__).resolve().parents[1]
    trainer = project_root / "scripts" / "train_dual_pi05.py"
    source_root = project_root / "src"
    openpi_source = Path(
        os.environ.get("PI05_OPENPI_SRC", project_root.parent / "openpi" / "src")
    )
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    if not openpi_source.is_dir():
        raise FileNotFoundError(openpi_source)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    processes: list[subprocess.Popen] = []
    logs = []
    try:
        for rank, gpu in enumerate(gpus):
            log_file = (args.log_dir / f"rank_{rank}.log").open("w", encoding="utf-8")
            logs.append(log_file)
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                    "XLA_FLAGS": (
                        f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0"
                    ).strip(),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": f"{source_root}:{openpi_source}",
                    "JAX_COMPILATION_CACHE_DIR": str(
                        Path(os.environ.get("PI05_JAX_CACHE_ROOT", "/tmp/pi05_fabric_jax_cache"))
                        / f"rank{rank}"
                    ),
                }
            )
            command = [
                sys.executable,
                str(trainer),
                *trainer_args,
                "--device-count",
                str(len(gpus)),
                "--distributed-process-count",
                str(len(gpus)),
                "--distributed-process-id",
                str(rank),
                "--distributed-coordinator-address",
                args.coordinator,
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=project_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            )

        while True:
            return_codes = [process.poll() for process in processes]
            failures = [code for code in return_codes if code not in (None, 0)]
            if failures:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError(
                    f"distributed training rank failed: return codes={return_codes}"
                )
            if all(code == 0 for code in return_codes):
                break
            time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for log_file in logs:
            log_file.close()

    print(f"DISTRIBUTED_TRAINING_COMPLETE log_dir={args.log_dir}")


if __name__ == "__main__":
    main()
