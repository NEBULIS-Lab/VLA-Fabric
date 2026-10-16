"""Pure contracts and JAX collectives for agent-parallel execution."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AgentGroupSpec:
    agent_count: int
    world_size: int
    agent_groups: tuple[tuple[int, ...], ...]
    role_data_parallel_groups: tuple[tuple[int, ...], ...]

    @classmethod
    def from_world_size(cls, *, world_size: int, agent_count: int) -> "AgentGroupSpec":
        if agent_count < 2:
            raise ValueError("agent_count must be at least two")
        if world_size <= 0 or world_size % agent_count:
            raise ValueError("world_size must be divisible by agent_count")
        group_count = world_size // agent_count
        agent_groups = tuple(
            tuple(range(group * agent_count, (group + 1) * agent_count))
            for group in range(group_count)
        )
        role_groups = tuple(
            tuple(group * agent_count + role for group in range(group_count))
            for role in range(agent_count)
        )
        return cls(agent_count, world_size, agent_groups, role_groups)


def role_ordered_peer_indices(
    receiver_role: int, *, agent_count: int, max_agents: int
) -> tuple[int, ...]:
    if not 0 <= receiver_role < agent_count:
        raise ValueError("receiver_role is outside the active team")
    if not agent_count <= max_agents:
        raise ValueError("max_agents cannot be smaller than agent_count")
    peers = [role for role in range(agent_count) if role != receiver_role]
    peers.extend([-1] * ((max_agents - 1) - len(peers)))
    return tuple(peers)


def pack_peer_contexts(
    contexts,
    *,
    receiver_role: int,
    max_agents: int,
):
    """Concatenate peer contexts in sender-role order and zero-fill spare slots."""

    values = jnp.asarray(contexts)
    if values.ndim < 2:
        raise ValueError("contexts must start with an agent axis")
    agent_count = values.shape[0]
    indices = role_ordered_peer_indices(
        receiver_role, agent_count=agent_count, max_agents=max_agents
    )
    zero = jnp.zeros_like(values[0])
    slots = [zero if index < 0 else values[index] for index in indices]
    packed = jnp.concatenate(slots, axis=-1)
    valid = jnp.asarray([index >= 0 for index in indices], dtype=jnp.bool_)
    return packed, valid


def select_role_ordered_peers(
    gathered,
    *,
    receiver_role,
    agent_count: int,
    max_agents: int,
):
    """Select peer tensors with a JAX-traceable receiver role.

    The leading axis of ``gathered`` must contain the active roles in role
    order. The result always has ``max_agents - 1`` slots, with absent slots
    zero-filled so one compiled graph supports the planned two-to-four-agent
    experiments.
    """

    values = jnp.asarray(gathered)
    if values.ndim < 1 or values.shape[0] != agent_count:
        raise ValueError("gathered values must have one leading slot per agent")
    table = jnp.asarray(
        [
            role_ordered_peer_indices(
                role, agent_count=agent_count, max_agents=max_agents
            )
            for role in range(agent_count)
        ],
        dtype=jnp.int32,
    )
    indices = table[jnp.asarray(receiver_role, dtype=jnp.int32)]
    valid = indices >= 0
    selected = values[jnp.maximum(indices, 0)]
    mask = valid
    while mask.ndim < selected.ndim:
        mask = mask[..., None]
    return jnp.where(mask, selected, jnp.zeros_like(selected)), valid


def reference_fuse_common(values, ownership, *, common_value: int):
    """Reference N-agent Common fusion used by CPU tests and audits."""

    values = jnp.asarray(values)
    owner = jnp.asarray(ownership)
    mean = jnp.mean(values, axis=0, keepdims=True)
    mean = jnp.broadcast_to(mean, values.shape)
    mask = owner == common_value
    if mask.ndim == 1:
        mask = mask[None]
    while mask.ndim < values.ndim - 1:
        mask = mask[..., None]
    mask = mask[None]
    return jnp.where(mask, mean, values)


def fuse_common_group(
    local_value,
    ownership,
    *,
    common_value: int,
    axis_name: str,
    axis_index_groups,
    agent_count: int,
):
    """Fuse Common-owned positions across one model-parallel agent group."""

    local_value = jnp.asarray(local_value)
    fused = jax.lax.psum(
        local_value, axis_name, axis_index_groups=axis_index_groups
    ) / agent_count
    mask = jnp.asarray(ownership) == common_value
    if mask.ndim == 1:
        mask = mask[None]
    while mask.ndim < local_value.ndim:
        mask = mask[..., None]
    return jnp.where(mask, fused, local_value)


def gather_agent_values(local_value, *, axis_name: str, axis_index_groups):
    """Gather the same tensor type from every role in one agent group."""

    return jax.lax.all_gather(
        local_value,
        axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=axis_index_groups,
    )
