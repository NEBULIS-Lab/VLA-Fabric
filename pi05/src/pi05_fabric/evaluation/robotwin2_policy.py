"""OpenPI-side policy runtime for RoboTwin2 requests."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Mapping

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils

from pi05_fabric.agents.base_provenance import fingerprint_base_checkpoint
from pi05_fabric.agents.base_provenance import validate_base_fingerprint
from pi05_fabric.agents.load_pi05 import load_dual_pi05
from pi05_fabric.agents.pi05_strong import native_trainable_filter
from pi05_fabric.data.aloha_dual_agent import merge_action
from pi05_fabric.data.aloha_dual_agent import split_observation
from pi05_fabric.data.converted_dataset import PairedTrainingSample
from pi05_fabric.data.pi05_batch import LocalQuantileNormalization
from pi05_fabric.data.pi05_batch import build_training_pair
from pi05_fabric.evaluation.native_handover import native_evaluation_spec
from pi05_fabric.evaluation.robotwin2_diagnostic import diagnostic_inference_mode
from pi05_fabric.evaluation.robotwin2_adapter import ACTION_HORIZON
from pi05_fabric.evaluation.robotwin2_adapter import validate_policy_payload
from pi05_fabric.evaluation.robotwin2_ipc import paired_flow_seed
from pi05_fabric.training.audit import audit_training_checkpoint
from pi05_fabric.training.checkpoint import restore_training_state
from pi05_fabric.training.stages import StageName


def validate_checkpoint_dataset(manifest: dict, dataset: Path) -> str:
    dataset = Path(dataset)
    raw = (dataset / 'manifest.json').read_bytes()
    statistics = json.loads((dataset / 'normalization_q01_q99.json').read_text())
    if statistics.get('dataset_manifest_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('policy dataset and normalization source hash differ')
    if manifest.get('protocol', {}).get('normalization') != statistics:
        raise ValueError('policy checkpoint normalization differs from supplied dataset')
    return json.loads(raw)['task']


def resolve_robotwin_evaluation_mode(stage: StageName, inference_profile: str | None):
    spec = native_evaluation_spec(stage)
    if inference_profile is None:
        return spec.mode
    if stage not in {
        StageName.PI_NATIVE_V2_RESIDUAL_ACTION_STAGE2,
        StageName.PI_NATIVE_V2_FULL_DIRECT,
        StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION,
    }:
        raise ValueError("diagnostic profile overrides require a V2 Full checkpoint")
    return diagnostic_inference_mode(inference_profile)


def policy_payload_to_pair(
    payload: Mapping[str, object],
    *,
    normalization: LocalQuantileNormalization,
    tokenizer,
):
    validated = validate_policy_payload(payload)
    sample = {
        "images": {
            "global": validated["global_image"],
            "left_wrist": validated["left_wrist_image"],
            "right_wrist": validated["right_wrist_image"],
        },
        "proprioception": validated["eef_state"],
        "instruction": str(validated["instruction"].item()),
    }
    left, right = split_observation(sample)
    zeros = np.zeros((ACTION_HORIZON, 10), dtype=np.float32)
    return build_training_pair(
        PairedTrainingSample(0, 0, left, right, zeros, zeros),
        normalization=normalization,
        tokenizer=tokenizer,
    )


def response_from_local_actions(
    *,
    request_id: str,
    flow_seed: int,
    left_actions,
    right_actions,
    normalization: LocalQuantileNormalization,
) -> dict[str, np.ndarray]:
    left, right = jax.device_get((left_actions, right_actions))
    left, right = normalization.unnormalize_actions(left, right)
    actions = merge_action(left[0], right[0]).astype(np.float32)
    if actions.shape != (ACTION_HORIZON, 20) or not np.isfinite(actions).all():
        raise FloatingPointError("policy returned an invalid RoboTwin action chunk")
    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "status": np.asarray("ok"),
        "request_id": np.asarray(request_id),
        "flow_seed": np.asarray(flow_seed, dtype=np.uint32),
        "actions_r6": actions,
    }


@dataclass
class RobotwinPi05Policy:
    sampler: object
    normalization: LocalQuantileNormalization
    tokenizer: object
    mode: object
    num_steps: int

    def infer(self, payload: Mapping[str, object]) -> dict[str, np.ndarray]:
        validated = validate_policy_payload(payload)
        condition_id = int(validated["condition_id"].item())
        planning_call = int(validated["planning_call"].item())
        request_id = str(validated["request_id"].item())
        seed = paired_flow_seed(condition_id, planning_call)
        pair = policy_payload_to_pair(
            validated, normalization=self.normalization, tokenizer=self.tokenizer
        )
        left_observation = jax.tree.map(jnp.asarray, pair.left_observation)
        right_observation = jax.tree.map(jnp.asarray, pair.right_observation)
        left, right = self.sampler(
            jax.random.key(seed),
            left_observation,
            right_observation,
            ownership=tuple(int(value) for value in pair.ownership[0]),
            mode=self.mode,
            num_steps=self.num_steps,
            preprocessed=False,
        )
        return response_from_local_actions(
            request_id=request_id,
            flow_seed=seed,
            left_actions=left,
            right_actions=right,
            normalization=self.normalization,
        )


def load_robotwin_pi05_policy(
    *,
    checkpoint: str | Path,
    expected_stage: str,
    expected_step: int,
    base_checkpoint: str | Path,
    dataset: str | Path,
    model_seed: int,
    num_steps: int,
    inference_profile: str | None = None,
) -> tuple[RobotwinPi05Policy, dict]:
    stage = StageName(expected_stage)
    spec = native_evaluation_spec(stage)
    mode = resolve_robotwin_evaluation_mode(stage, inference_profile)
    if spec.action_horizon != ACTION_HORIZON:
        raise ValueError("RoboTwin2 policy requires an H50 V2 checkpoint")
    manifest = audit_training_checkpoint(
        checkpoint, expected_stage=stage.value, expected_step=expected_step,
        verify_model=stage not in (
            StageName.PI_NATIVE_V2_FULL_DIRECT,
            StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION,
        ),
    )
    validate_checkpoint_dataset(manifest, Path(dataset))
    if int(manifest.get("model_seed", -1)) != model_seed:
        raise ValueError("checkpoint model seed does not match policy server")
    if stage in (
        StageName.PI_NATIVE_V2_FULL_DIRECT,
        StageName.PI_NATIVE_V2_EXPANDED_CONTINUATION,
    ):
        base_fingerprint = fingerprint_base_checkpoint(base_checkpoint)
        validate_base_fingerprint(manifest.get("protocol"), base_fingerprint["tree_sha256"])
    # Direct defers tensor/model-hash verification until its frozen base is verified.
    snapshot = restore_training_state(checkpoint)
    _, model = load_dual_pi05(
        base_checkpoint,
        seed=model_seed,
        mode=mode,
        train_action_ffw=spec.train_action_ffw,
        train_action_attention=spec.train_action_attention,
        train_paligemma_kv=spec.train_paligemma_kv,
        train_paligemma_qo=spec.train_paligemma_qo,
        separate_expanded_groups=spec.separate_expanded_groups,
        action_horizon=ACTION_HORIZON,
    )
    selected = nnx.state(model).filter(
        native_trainable_filter(
            train_action_ffw=spec.train_action_ffw,
            train_action_attention=spec.train_action_attention,
            train_paligemma_kv=spec.train_paligemma_kv,
            train_paligemma_qo=spec.train_paligemma_qo,
            separate_expanded_groups=spec.separate_expanded_groups,
        )
    )
    selected.replace_by_pure_dict(snapshot.params)
    nnx.update(model, selected)
    del selected, snapshot
    gc.collect()
    sampler = nnx_utils.module_jit(
        model.sample_actions,
        static_argnames=("ownership", "mode", "num_steps", "preprocessed"),
    )
    normalization = LocalQuantileNormalization.from_json(
        Path(dataset) / "normalization_q01_q99.json"
    )
    return (
        RobotwinPi05Policy(sampler, normalization, PaligemmaTokenizer(200), mode, num_steps),
        manifest,
    )
