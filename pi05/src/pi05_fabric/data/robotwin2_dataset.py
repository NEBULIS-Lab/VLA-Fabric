"""RoboTwin2-specific audits for the shared paired-trajectory archive format."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


EXPECTED_TASK = "robotwin_handover_block"
EXPECTED_EPISODES = 50
EXPECTED_STEPS = 14084
LOCAL_DIM = 10
JOINT_DIM = 20

# None means that the step count must come from a full RLDS source scan.
TASK_EXPECTED_STEPS = {
    EXPECTED_TASK: EXPECTED_STEPS,
    "robotwin_handover_mic": None,
    "robotwin_hanging_mug": None,
    "robotwin_scan_object": None,
}

_FIELDS = (
    "global_image",
    "left_wrist_image",
    "right_wrist_image",
    "proprioception",
    "action",
    "instruction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_steps(task: str) -> int | None:
    if task not in TASK_EXPECTED_STEPS:
        raise ValueError(f"unsupported RoboTwin2 task: {task!r}")
    return TASK_EXPECTED_STEPS[task]


def validate_source_metadata(
    dataset_dir: str | Path, *, task: str = EXPECTED_TASK,
) -> dict[str, object]:
    _expected_steps(task)
    dataset_dir = Path(dataset_dir)
    info = json.loads((dataset_dir / "dataset_info.json").read_text())
    if info.get("name") != task:
        raise ValueError(f"expected source task {task!r}, got {info.get('name')!r}")
    try:
        split = next(item for item in info["splits"] if item["name"] == "train")
    except StopIteration as exc:
        raise ValueError("source dataset has no train split") from exc
    shard_lengths = tuple(int(value) for value in split["shardLengths"])
    if len(shard_lengths) != 8 or sum(shard_lengths) != EXPECTED_EPISODES:
        raise ValueError(
            f"expected eight shards and {EXPECTED_EPISODES} episodes, got "
            f"{len(shard_lengths)} shards and {sum(shard_lengths)} episodes"
        )
    return {
        "task": info["name"],
        "episode_count": sum(shard_lengths),
        "shard_count": len(shard_lengths),
        "declared_bytes": int(split["numBytes"]),
    }


def _update_step_digest(digest, arrays: dict[str, np.ndarray]) -> None:
    """Fingerprint aligned fields, including arm order, in conversion precision."""
    for field in _FIELDS:
        value = np.asarray(arrays[field])
        if field == "instruction":
            if value.ndim != 1 or not len(value):
                raise ValueError("source instruction list must be non-empty")
            strings = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value]
            if any(not item.strip() for item in strings):
                raise ValueError("empty source instruction")
            payload = json.dumps(strings, ensure_ascii=True).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
            continue
        if field in ("proprioception", "action"):
            value = value.astype(np.float32)
            if value.shape != (JOINT_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"invalid or non-finite source {field}")
        elif value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"invalid source {field}")
        header = json.dumps([field, value.dtype.str, value.shape]).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(np.ascontiguousarray(value).tobytes())


def audit_source_dataset(
    dataset_dir: str | Path, *, task: str = EXPECTED_TASK,
) -> dict[str, object]:
    """Scan every training episode; metadata episode counts are not step counts.

    Requires TensorFlow/TFDS and is intended for a Slurm CPU conversion job.
    The ordinary converted-data audit remains TensorFlow-free.
    """
    from pi05_fabric.data.converted_dataset import expected_rlds_shards

    dataset_dir = Path(dataset_dir)
    metadata = validate_source_metadata(dataset_dir, task=task)
    shards = expected_rlds_shards(dataset_dir)
    declared_shards = [
        {"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        for path in shards
    ]
    info_hash = sha256(dataset_dir / "dataset_info.json")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir.resolve()))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    episodes = []
    for episode_id, raw_episode in enumerate(tfds.as_numpy(dataset)):
        digest = hashlib.sha256()
        length = 0
        instruction = None
        for step in raw_episode["steps"]:
            arrays = {
                "global_image": step["observation"]["image"],
                "left_wrist_image": step["observation"]["left_wrist_image"],
                "right_wrist_image": step["observation"]["right_wrist_image"],
                "proprioception": step["observation"]["eef_state"],
                "action": step["eef_action"],
                "instruction": step["language_instruction"],
            }
            _update_step_digest(digest, arrays)
            current_instruction = np.asarray(arrays["instruction"])
            if instruction is not None and not np.array_equal(instruction, current_instruction):
                raise ValueError(f"inconsistent source instruction in episode {episode_id}")
            instruction = current_instruction.copy()
            length += 1
        if not length:
            raise ValueError(f"source episode {episode_id} has no steps")
        source_file_path = raw_episode.get("episode_metadata", {}).get("file_path")
        if isinstance(source_file_path, np.ndarray) and source_file_path.ndim == 0:
            source_file_path = source_file_path.item()
        if isinstance(source_file_path, bytes):
            source_file_path = source_file_path.decode("utf-8")
        if source_file_path is not None and not isinstance(source_file_path, str):
            raise ValueError(f"invalid source episode_metadata/file_path in episode {episode_id}")
        episodes.append({
            "id": episode_id,
            "length": length,
            "content_sha256": digest.hexdigest(),
            "source_file_path": source_file_path,
        })
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} source episodes, got {len(episodes)}")
    total_steps = sum(item["length"] for item in episodes)
    expected_steps = _expected_steps(task)
    if expected_steps is not None and total_steps != expected_steps:
        raise ValueError(f"expected {expected_steps} source steps, got {total_steps}")
    return {
        "schema_version": 1,
        "status": "PASS",
        **metadata,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_info_sha256": info_hash,
        "split": "train",
        "declared_shards": declared_shards,
        "episodes": episodes,
        "step_count": total_steps,
        "replay_provenance": {
            "episode_id_semantics": "TFDS train iteration index, not an environment seed",
            "source_file_path_field": "episode_metadata/file_path",
            "seed_mapping_verified": False,
            "initial_state_reconstruction": "not_verified",
        },
    }


def _conversion_source_audit(root: Path, manifest: dict, *, task: str) -> dict:
    path = root / "robotwin2_source_audit.json"
    if not path.is_file():
        raise ValueError(f"{task} requires a full RLDS source audit: {path}")
    source = json.loads(path.read_text())
    if (source.get("schema_version") != 1 or source.get("status") != "PASS"
            or source.get("task") != task or source.get("split") != "train"):
        raise ValueError("source audit schema, status, task or split mismatch")
    episodes = source.get("episodes", [])
    if (source.get("episode_count") != EXPECTED_EPISODES or len(episodes) != EXPECTED_EPISODES
            or source.get("shard_count") != 8 or len(source.get("declared_shards", [])) != 8):
        raise ValueError("source audit must cover all 50 episodes and eight shards")
    if ([item.get("id") for item in episodes] != list(range(EXPECTED_EPISODES))
            or any(type(item.get("length")) is not int or item["length"] <= 0 for item in episodes)):
        raise ValueError("source audit has invalid episode IDs or lengths")
    if source.get("step_count") != sum(item["length"] for item in episodes):
        raise ValueError("source audit step count disagrees with source episode lengths")
    manifest_source = manifest.get("source", {})
    for field in ("dataset_dir", "split", "declared_shards"):
        if manifest_source.get(field) != source.get(field):
            raise ValueError(f"converted/source audit mismatch: {field}")
    return source


def audit_converted_dataset(
    root: str | Path,
    *,
    require_full_dataset: bool,
    task: str = EXPECTED_TASK,
) -> dict[str, object]:
    expected_steps = _expected_steps(task)
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported converted dataset schema")
    if manifest.get("task") != task:
        raise ValueError(f"converted task mismatch: {manifest.get('task')!r}")

    episodes = manifest.get("episodes", [])
    if require_full_dataset and len(episodes) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} converted episodes")

    source = None
    if expected_steps is None:
        source = _conversion_source_audit(root, manifest, task=task)
        expected_steps = source["step_count"]
        if (len(episodes) > EXPECTED_EPISODES
                or [item.get("id") for item in episodes] != list(range(len(episodes)))
                or len({item["path"] for item in episodes}) != len(episodes)):
            raise ValueError("converted/source episode IDs mismatch or duplicate paths")

    total_steps = 0
    instructions: set[str] = set()
    image_shapes: dict[str, set[tuple[int, ...]]] = {
        "global_image": set(),
        "left_wrist_image": set(),
        "right_wrist_image": set(),
    }
    for item in episodes:
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"episode hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = set(_FIELDS) - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            state = np.asarray(archive["proprioception"])
            action = np.asarray(archive["action"])
            if state.ndim != 2 or state.shape[1] != JOINT_DIM:
                raise ValueError(f"invalid state shape in {path}: {state.shape}")
            if action.shape != state.shape:
                raise ValueError(f"state/action shape mismatch in {path}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"non-finite state/action in {path}")
            length = state.shape[0]
            if length != int(item["length"]):
                raise ValueError(f"manifest length mismatch in {path}")
            for field in image_shapes:
                images = np.asarray(archive[field])
                if images.shape[0] != length or images.ndim != 4 or images.shape[-1] != 3:
                    raise ValueError(f"invalid {field} shape in {path}: {images.shape}")
                if images.dtype != np.uint8:
                    raise ValueError(f"{field} must be uint8 in {path}")
                image_shapes[field].add(tuple(images.shape[1:]))
            instruction_array = np.asarray(archive["instruction"])
            if instruction_array.ndim != 1 or not len(instruction_array):
                raise ValueError(f"instruction list must be non-empty in {path}")
            episode_instructions = {str(item).strip() for item in instruction_array}
            if not episode_instructions or "" in episode_instructions:
                raise ValueError(f"empty instruction in {path}")
            instructions.update(episode_instructions)
            if source is not None:
                source_episode = source["episodes"][item["id"]]
                if length != source_episode["length"]:
                    raise ValueError(f"converted/source episode length mismatch in {path}")
                digest = hashlib.sha256()
                arrays = {field: np.asarray(archive[field]) for field in _FIELDS}
                for index in range(length):
                    _update_step_digest(digest, {
                        field: value if field == "instruction" else value[index]
                        for field, value in arrays.items()
                    })
                if digest.hexdigest() != source_episode.get("content_sha256"):
                    raise ValueError(f"converted/source content mismatch in {path}")
            total_steps += length

    if require_full_dataset and total_steps != expected_steps:
        raise ValueError(f"expected {expected_steps} converted steps, got {total_steps}")
    if not episodes:
        raise ValueError("converted dataset has no episodes")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "task": task,
        "manifest_sha256": sha256(manifest_path),
        "episode_count": len(episodes),
        "step_count": total_steps,
        "joint_dim": JOINT_DIM,
        "local_dim": LOCAL_DIM,
        "instruction_count": len(instructions),
        "image_shapes": {
            field: [list(shape) for shape in sorted(shapes)]
            for field, shapes in image_shapes.items()
        },
    }
    if source is not None:
        result.update({
            "source_step_count": source["step_count"],
            "source_audit_sha256": sha256(root / "robotwin2_source_audit.json"),
            "full_dataset": len(episodes) == EXPECTED_EPISODES,
            "instructions": sorted(instructions),
        })
    return result
