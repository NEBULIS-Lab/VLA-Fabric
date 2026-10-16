"""Helpers for one-process-per-GPU synchronous data parallelism."""

from __future__ import annotations

import jax


def validate_replica_topology(
    *,
    requested_device_count: int,
    process_count: int,
    local_device_count: int,
) -> None:
    if requested_device_count <= 0:
        raise ValueError("requested_device_count must be positive")
    if requested_device_count == 1:
        if process_count != 1 or local_device_count != 1:
            raise ValueError("single-device training requires one process and one local GPU")
        return
    if process_count != requested_device_count or local_device_count != 1:
        raise ValueError(
            "multi-GPU training requires one process per GPU: "
            f"requested={requested_device_count}, processes={process_count}, "
            f"local_devices={local_device_count}"
        )


def process_sample_indices(*, process_index: int, samples_per_process: int) -> tuple[int, ...]:
    if process_index < 0:
        raise ValueError("process_index must be non-negative")
    if samples_per_process <= 0:
        raise ValueError("samples_per_process must be positive")
    start = process_index * samples_per_process
    return tuple(range(start, start + samples_per_process))


def process_device_rngs(
    rng,
    *,
    process_index: int,
    process_count: int,
    local_device_count: int,
):
    if process_count <= 0 or local_device_count <= 0:
        raise ValueError("process_count and local_device_count must be positive")
    if not 0 <= process_index < process_count:
        raise ValueError("process_index is outside the distributed world")
    all_keys = jax.random.split(rng, process_count * local_device_count)
    start = process_index * local_device_count
    return all_keys[start : start + local_device_count]


def shard_leading_axis(value, *, device_count: int):
    if device_count <= 0:
        raise ValueError("device_count must be positive")
    if value.shape[0] % device_count != 0:
        raise ValueError(
            f"local batch {value.shape[0]} is not divisible by {device_count} local devices"
        )
    local_batch = value.shape[0] // device_count
    return value.reshape((device_count, local_batch, *value.shape[1:]))


def shard_batch(tree, *, device_count: int):
    return jax.tree.map(
        lambda value: shard_leading_axis(value, device_count=device_count),
        tree,
    )


def unreplicate_tree(tree):
    return jax.tree.map(lambda value: value[0], tree)
