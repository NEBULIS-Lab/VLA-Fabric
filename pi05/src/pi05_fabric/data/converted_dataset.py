"""TensorFlow-free reader for converted paired ALOHA trajectories."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from pi05_fabric.data.aloha_dual_agent import AgentObservation
from pi05_fabric.data.aloha_dual_agent import JOINT_DIM
from pi05_fabric.data.aloha_dual_agent import split_action
from pi05_fabric.data.aloha_dual_agent import split_observation


_EPISODE_FIELDS = (
    "global_image",
    "left_wrist_image",
    "right_wrist_image",
    "proprioception",
    "action",
    "instruction",
)


@dataclass(frozen=True)
class PairedTrainingSample:
    episode_id: int
    step: int
    left: AgentObservation
    right: AgentObservation
    left_action: np.ndarray
    right_action: np.ndarray


def expected_rlds_shards(dataset_dir: str | Path, *, split: str = "train") -> tuple[Path, ...]:
    """Resolve exactly the shard set declared by TFDS metadata."""

    dataset_dir = Path(dataset_dir)
    info = json.loads((dataset_dir / "dataset_info.json").read_text())
    try:
        split_info = next(item for item in info["splits"] if item["name"] == split)
    except StopIteration as exc:
        raise ValueError(f"split {split!r} is absent from dataset_info.json") from exc
    count = len(split_info["shardLengths"])
    name = info["name"]
    shards = tuple(
        dataset_dir / f"{name}-{split}.tfrecord-{index:05d}-of-{count:05d}" for index in range(count)
    )
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(f"missing declared RLDS shard: {shard.name}")
    return shards


def build_action_chunk(actions: np.ndarray, *, start: int, horizon: int) -> np.ndarray:
    """Slice a fixed horizon and repeat the terminal command when needed."""

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[-1] != JOINT_DIM:
        raise ValueError(f"actions must have shape [time, {JOINT_DIM}]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= start < len(actions):
        raise IndexError(f"step {start} is outside an episode of length {len(actions)}")
    stop = min(start + horizon, len(actions))
    chunk = actions[start:stop]
    if len(chunk) < horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)], axis=0)
    return chunk


class ConvertedAlohaDataset:
    """Random-access paired samples from immutable episode archives."""

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int = 20,
        preload: bool = False,
        preload_workers: int = 1,
    ):
        if preload_workers <= 0:
            raise ValueError("preload_workers must be positive")
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported converted-dataset schema")
        self._episodes = {int(item["id"]): item for item in self.manifest["episodes"]}
        self._index = tuple(
            (episode_id, step)
            for episode_id, item in self._episodes.items()
            for step in range(int(item["length"]))
        )
        self._episode_cache: dict[int, dict[str, np.ndarray]] = {}
        if preload:
            episode_ids = tuple(sorted(self._episodes))
            if preload_workers == 1:
                episodes = map(self._read_episode, episode_ids)
                self._episode_cache = dict(zip(episode_ids, episodes, strict=True))
            else:
                with ThreadPoolExecutor(max_workers=preload_workers) as executor:
                    episodes = executor.map(self._read_episode, episode_ids)
                    self._episode_cache = dict(
                        zip(episode_ids, episodes, strict=True)
                    )

    def __len__(self) -> int:
        return len(self._index)

    def indices(self) -> Iterator[tuple[int, int]]:
        return iter(self._index)

    def sample_index(self, rng: np.random.Generator) -> tuple[int, int]:
        return self._index[int(rng.integers(len(self._index)))]

    @property
    def preloaded_bytes(self) -> int:
        return sum(
            array.nbytes
            for episode in self._episode_cache.values()
            for array in episode.values()
        )

    def _read_episode(self, episode_id: int) -> dict[str, np.ndarray]:
        item = self._episodes[episode_id]
        archive_path = self.root / item["path"]
        with np.load(archive_path, allow_pickle=False) as episode:
            arrays = {name: np.asarray(episode[name]) for name in _EPISODE_FIELDS}
        length = len(arrays["action"])
        if length != int(item["length"]):
            raise ValueError(
                f"episode {episode_id} manifest length {item['length']} does not match archive length {length}"
            )
        for array in arrays.values():
            array.setflags(write=False)
        return arrays

    def get(self, episode_id: int, step: int) -> PairedTrainingSample:
        episode = self._episode_cache.get(episode_id)
        if episode is None:
            episode = self._read_episode(episode_id)
        instruction = np.asarray(episode["instruction"])
        if instruction.ndim == 0:
            selected_instruction = str(instruction.item())
        elif instruction.ndim == 1 and len(instruction):
            # RoboTwin stores all seen paraphrases once per episode. A stable
            # index preserves instruction diversity without rank-local RNG.
            selected_instruction = str(instruction[(episode_id + step) % len(instruction)])
        else:
            raise ValueError(
                f"episode {episode_id} instruction must be a scalar or non-empty vector"
            )
        sample = {
            "images": {
                "global": episode["global_image"][step],
                "left_wrist": episode["left_wrist_image"][step],
                "right_wrist": episode["right_wrist_image"][step],
            },
            "proprioception": episode["proprioception"][step],
            "instruction": selected_instruction,
        }
        left, right = split_observation(sample)
        left_action, right_action = split_action(
            build_action_chunk(episode["action"], start=step, horizon=self.action_horizon)
        )
        return PairedTrainingSample(episode_id, step, left, right, left_action, right_action)
