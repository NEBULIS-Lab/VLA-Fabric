"""Static topology and deterministic RNG contracts for agent parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from pi05_fabric.communication.multiagent_collectives import AgentGroupSpec


@dataclass(frozen=True)
class BatchConfiguration:
    team_microbatch: int
    accumulation: int
    data_parallel_teams: int
    global_team_batch: int


def batch_configuration(
    *,
    team_microbatch: int,
    accumulation: int,
    data_parallel_teams: int,
) -> BatchConfiguration:
    if min(team_microbatch, accumulation, data_parallel_teams) <= 0:
        raise ValueError("batch factors must be positive")
    return BatchConfiguration(
        team_microbatch=team_microbatch,
        accumulation=accumulation,
        data_parallel_teams=data_parallel_teams,
        global_team_batch=team_microbatch * accumulation * data_parallel_teams,
    )


@dataclass(frozen=True)
class AgentParallelTopology:
    world_size: int
    agent_count: int
    agent_groups: tuple[tuple[int, ...], ...]
    role_data_parallel_groups: tuple[tuple[int, ...], ...]

    @classmethod
    def create(cls, *, world_size: int, agent_count: int) -> "AgentParallelTopology":
        groups = AgentGroupSpec.from_world_size(
            world_size=world_size,
            agent_count=agent_count,
        )
        return cls(
            world_size=world_size,
            agent_count=agent_count,
            agent_groups=groups.agent_groups,
            role_data_parallel_groups=groups.role_data_parallel_groups,
        )

    @property
    def team_count(self) -> int:
        return self.world_size // self.agent_count

    def role_for_rank(self, rank: int) -> int:
        self._validate_rank(rank)
        return rank % self.agent_count

    def team_for_rank(self, rank: int) -> int:
        self._validate_rank(rank)
        return rank // self.agent_count

    def _validate_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the distributed world")


def team_rng(*, seed: int, step: int, microstep: int, team_index: int):
    """Return one key shared by all roles of a team for one microbatch."""

    if min(step, microstep, team_index) < 0:
        raise ValueError("RNG coordinates must be non-negative")
    key = jax.random.key(seed)
    key = jax.random.fold_in(key, step)
    key = jax.random.fold_in(key, microstep)
    return jax.random.fold_in(key, team_index)


def role_rng(
    *, seed: int, step: int, microstep: int, team_index: int, role_index: int
):
    """Return a deterministic flow-noise key private to one agent role."""

    if role_index < 0:
        raise ValueError("role_index must be non-negative")
    return jax.random.fold_in(
        team_rng(
            seed=seed,
            step=step,
            microstep=microstep,
            team_index=team_index,
        ),
        role_index,
    )
