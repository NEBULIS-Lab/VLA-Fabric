"""Train CommVLA-native v3 on dual-arm RLDS data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names

if TYPE_CHECKING:
    from commvla.models.dual_arm import CommVLANativeV3Pair


def add_project_paths(project_root: Path) -> None:
    paths = [Path(__file__).resolve().parents[2]]
    if os.environ.get("TWINVLA_ROOT"):
        paths.append(Path(os.environ["TWINVLA_ROOT"]).expanduser())
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--singlevla-pretrained-path", required=True)
    parser.add_argument("--native-checkpoint", default=None, help="Optional full CommVLA-native checkpoint to initialize from.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume model, optimizer, scheduler, global step, and per-rank RNG from a training checkpoint.",
    )
    parser.add_argument("--left-agent-path", default=None)
    parser.add_argument("--right-agent-path", default=None)
    parser.add_argument("--right-device", default=None)
    parser.add_argument("--data-root-dir", required=True)
    parser.add_argument("--data-mix", default="aloha_handover_box")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="Stop cleanly after this optimizer step; max_steps and scheduler horizon remain unchanged.",
    )
    parser.add_argument(
        "--skip-save-on-stop-after",
        action="store_true",
        help="For smoke tests only: stop cleanly without writing a checkpoint at --stop-after-step.",
    )
    parser.add_argument(
        "--checkpoint-request-file",
        default=None,
        help="If this shared file exists after an optimizer step, save a resumable checkpoint and stop cleanly.",
    )
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument(
        "--save-at-steps",
        default=None,
        help="Optional comma-separated optimizer steps to save; overrides periodic --save-steps.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--vision-learning-rate", type=float, default=None)
    parser.add_argument("--llm-learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--action-transformer-learning-rate", type=float, default=None)
    parser.add_argument("--action-decoder-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-parallel-calls", type=int, default=8)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--freeze-vision-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision-backbone", dest="freeze_vision_backbone", action="store_false")
    parser.add_argument("--freeze-llm-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-llm-backbone", dest="freeze_llm_backbone", action="store_false")
    parser.add_argument("--train-llm-last-n-layers", type=int, default=0)
    parser.add_argument("--remote-kv-mode", choices=["full", "no_remote", "random_remote"], default="full")
    parser.add_argument("--common-strategy", choices=["local_common"], default="local_common")
    parser.add_argument(
        "--common-kv-mode",
        choices=["local", "left_shared", "right_shared", "avg_shared", "sync_qkv_avg_mlp_avg"],
        default="local",
    )
    parser.add_argument("--common-fusion-mode", choices=["fixed_avg", "layer_gate"], default="fixed_avg")
    parser.add_argument(
        "--action-token-fusion-mode",
        choices=["none", "residual_exchange", "paired_concat_project", "paired_residual_mlp", "paired_joint_transformer"],
        default="none",
    )
    parser.add_argument("--action-transformer-layers", type=int, default=3)
    parser.add_argument("--action-transformer-heads", type=int, default=8)
    parser.add_argument("--action-transformer-mlp-ratio", type=int, default=2)
    parser.add_argument("--decoder-mode", choices=["separate_decoders", "shared_decoder_batched"], default="separate_decoders")
    parser.add_argument("--loss-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--remote-kv-scale", type=float, default=1.0)
    parser.add_argument("--remote-kv-dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--train-scope",
        choices=[
            "default",
            "action_fusion_only",
            "action_fusion_residual_only",
            "action_fusion_head_boundary",
            "action_fusion_full_action_head",
            "joint_transformer_only",
            "joint_transformer_action_decoder",
        ],
        default="default",
    )
    parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--ddp-timeout-minutes", type=int, default=30)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true", default=True)
    parser.add_argument("--no-ddp-find-unused-parameters", dest="ddp_find_unused_parameters", action="store_false")
    parser.add_argument(
        "--require-zero-model-communication",
        action="store_true",
        help="Fail after the first forward unless Common, Private K/V, and Action communication counters are all zero.",
    )
    parser.add_argument(
        "--require-private-only-model-communication",
        action="store_true",
        help="Fail after the first forward unless Private K/V is nonzero and Common and Action communication are zero.",
    )
    parser.add_argument(
        "--require-expected-model-communication",
        action="store_true",
        help="Validate enabled communication groups from the effective model configuration on the first forward.",
    )
    parser.add_argument(
        "--interaction-curriculum",
        choices=["none", "common_only_then_stage2"],
        default="none",
        help="Optional step-based communication curriculum applied without restarting optimization.",
    )
    parser.add_argument(
        "--curriculum-transition-step",
        type=int,
        default=None,
        help="Last optimizer step in Stage1; the next step uses the native Stage2 communication modes.",
    )
    parser.add_argument(
        "--curriculum-stage1-common-kv-mode",
        choices=["avg_shared", "sync_qkv_avg_mlp_avg"],
        default="avg_shared",
        help="Symmetric Common-only mode used before the curriculum transition.",
    )
    parser.add_argument(
        "--require-staged-model-communication",
        action="store_true",
        help="Validate Common-only Stage1 and the native Stage2 communication profile at their first forwards.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state(device: str) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return state


def curriculum_phase_for_step(args: argparse.Namespace, optimizer_step: int) -> str:
    if args.interaction_curriculum == "none":
        return "stage2"
    if args.curriculum_transition_step is None:
        raise ValueError("--curriculum-transition-step is required for a staged curriculum")
    return "stage1_common_only" if optimizer_step <= args.curriculum_transition_step else "stage2"


def apply_curriculum_phase(model, args: argparse.Namespace, phase: str) -> dict[str, str]:
    model.clear_runtime_communication_modes()
    if phase == "stage1_common_only":
        model.set_runtime_communication_modes(
            common_kv_mode=args.curriculum_stage1_common_kv_mode,
            private_remote_kv_mode="no_remote",
            action_token_ablation_mode="disable_fusion",
        )
        return {
            "phase": phase,
            "common_kv_mode": args.curriculum_stage1_common_kv_mode,
            "private_remote_kv_mode": "no_remote",
            "action_token_mode": "disable_fusion",
        }
    return {
        "phase": phase,
        "common_kv_mode": args.common_kv_mode,
        "private_remote_kv_mode": args.remote_kv_mode,
        "action_token_mode": args.action_token_fusion_mode,
    }


def expected_communication_groups(args: argparse.Namespace, phase: str) -> dict[str, bool]:
    if phase == "stage1_common_only":
        return {"common": True, "private_kv": False, "action_intent": False}
    return {
        "common": args.common_kv_mode != "local",
        "private_kv": args.remote_kv_mode != "no_remote",
        "action_intent": args.action_token_fusion_mode != "none",
    }


def validate_communication_groups(
    communication_profile: dict[str, int],
    expected: dict[str, bool],
    *,
    label: str,
) -> None:
    mismatches = {
        group: {
            "expected_nonzero": should_be_nonzero,
            "observed_bytes": communication_profile[group],
        }
        for group, should_be_nonzero in expected.items()
        if (communication_profile[group] > 0) != should_be_nonzero
    }
    expected_total = sum(
        communication_profile[group] for group, enabled in expected.items() if enabled
    )
    if communication_profile["total"] != expected_total:
        mismatches["total"] = {
            "expected_bytes": expected_total,
            "observed_bytes": communication_profile["total"],
        }
    if mismatches:
        raise RuntimeError(
            f"Unexpected {label} model communication: "
            + json.dumps(
                {"profile": communication_profile, "mismatches": mismatches},
                sort_keys=True,
            )
        )


def restore_rng_state(state: dict, device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], torch.device(device))


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int, str]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend=args.dist_backend,
            timeout=timedelta(minutes=args.ddp_timeout_minutes),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return rank, world_size, local_rank, device


def build_training_arg_shims(args: argparse.Namespace):
    from scripts.configs.twinvla_config import ModelArguments, TrainingArguments

    model_args = ModelArguments(
        model_type="Eagle2_1BVLA",
        singlevla_pretrained_path=args.singlevla_pretrained_path,
        action_len=20,
        normalization="quantile",
        global_normalization=True,
        hz_interpolate=None,
        interpolate_gripper=False,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_workers=args.num_workers,
        enable_autotune=False,
        num_parallel_calls=args.num_parallel_calls,
        image_aug=False,
    )
    return model_args, training_args


def trainable_parameter_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    *,
    vision_learning_rate: float | None = None,
    llm_learning_rate: float | None = None,
    head_learning_rate: float | None = None,
    action_transformer_learning_rate: float | None = None,
    action_decoder_learning_rate: float | None = None,
):
    vision_lr = learning_rate if vision_learning_rate is None else vision_learning_rate
    llm_lr = learning_rate if llm_learning_rate is None else llm_learning_rate
    head_lr = learning_rate if head_learning_rate is None else head_learning_rate
    action_transformer_lr = head_lr if action_transformer_learning_rate is None else action_transformer_learning_rate
    action_decoder_lr = head_lr if action_decoder_learning_rate is None else action_decoder_learning_rate

    base_model = model.module if isinstance(model, DDP) else model

    def param_ids(module: nn.Module | None) -> set[int]:
        if module is None:
            return set()
        return {id(param) for param in module.parameters()}

    vision_ids: set[int] = set()
    llm_ids: set[int] = set()
    head_ids: set[int] = set()
    action_transformer_ids: set[int] = set()
    action_decoder_ids: set[int] = set()
    for agent in (base_model.left.model, base_model.right.model):
        vision_ids.update(param_ids(agent.vision_backbone()))
        llm_ids.update(param_ids(agent.text_backbone()))
        for module_name in ("embed_arm_state", "agg"):
            head_ids.update(param_ids(getattr(agent, module_name, None)))
        action_decoder_ids.update(param_ids(getattr(agent, "agg", None)))
        action_token = getattr(agent, "action_token", None)
        if action_token is not None:
            head_ids.add(id(action_token))
            action_decoder_ids.add(id(action_token))
    if base_model.native_config.decoder_mode == "shared_decoder_batched":
        head_ids.update(param_ids(base_model.shared_action_head))
    else:
        head_ids.update(param_ids(base_model.left.model.action_head))
        head_ids.update(param_ids(base_model.right.model.action_head))
        action_decoder_ids.update(param_ids(base_model.left.model.action_head))
        action_decoder_ids.update(param_ids(base_model.right.model.action_head))
    if getattr(base_model, "common_layer_gate", None) is not None:
        head_ids.add(id(base_model.common_layer_gate))
    head_ids.update(param_ids(getattr(base_model, "left_action_token_fusion", None)))
    head_ids.update(param_ids(getattr(base_model, "right_action_token_fusion", None)))
    head_ids.update(param_ids(getattr(base_model, "joint_action_token_transformer", None)))
    action_transformer_ids.update(param_ids(getattr(base_model, "joint_action_token_transformer", None)))

    def lr_bucket(param: nn.Parameter) -> str:
        param_id = id(param)
        if param_id in action_transformer_ids:
            return "action_transformer"
        if param_id in action_decoder_ids:
            return "action_decoder"
        if param_id in head_ids:
            return "head"
        if param_id in vision_ids:
            return "vision"
        if param_id in llm_ids:
            return "llm"
        return "llm"

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_names = {name for name in decay_parameters if "bias" not in name}
    bucket_lrs = {
        "vision": vision_lr,
        "llm": llm_lr,
        "head": head_lr,
        "action_transformer": action_transformer_lr,
        "action_decoder": action_decoder_lr,
    }
    groups = []
    for bucket in ("vision", "llm", "head", "action_transformer", "action_decoder"):
        lr = bucket_lrs[bucket]
        groups.append(
            {
                "params": [p for n, p in model.named_parameters() if n in decay_names and p.requires_grad and lr_bucket(p) == bucket],
                "weight_decay": weight_decay,
                "lr": lr,
                "name": f"{bucket}_decay",
            }
        )
        groups.append(
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_names and p.requires_grad and lr_bucket(p) == bucket],
                "weight_decay": 0.0,
                "lr": lr,
                "name": f"{bucket}_no_decay",
            }
        )
    grouped = [group for group in groups if group["params"]]
    summary = {}
    for bucket in ("vision", "llm", "head", "action_transformer", "action_decoder"):
        params = [p for p in model.parameters() if p.requires_grad and lr_bucket(p) == bucket]
        summary[bucket] = {
            "params": sum(p.numel() for p in params),
            "tensors": len(params),
            "lr": bucket_lrs[bucket],
        }
    return grouped, summary


def apply_train_scope(model: CommVLANativeV3Pair, scope: str) -> dict:
    if scope == "default":
        return {"scope": scope, "changed": False}

    for param in model.parameters():
        param.requires_grad = False

    enabled_modules: list[str] = []

    def enable_module(name: str, module: nn.Module | None) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = True
        enabled_modules.append(name)

    def enable_parameter(name: str, parameter: nn.Parameter | None) -> None:
        if parameter is None:
            return
        parameter.requires_grad = True
        enabled_modules.append(name)

    if scope in ("joint_transformer_only", "joint_transformer_action_decoder"):
        transformer = getattr(model, "joint_action_token_transformer", None)
        if transformer is None:
            raise RuntimeError(f"{scope} requires joint_action_token_transformer")
        enable_module("joint_action_token_transformer", transformer)
        if scope == "joint_transformer_action_decoder":
            for side_name, agent in (("left", model.left.model), ("right", model.right.model)):
                enable_module(f"{side_name}_action_head", agent.action_head)
                enable_module(f"{side_name}_agg", getattr(agent, "agg", None))
                enable_parameter(f"{side_name}_action_token", getattr(agent, "action_token", None))
        trainable_tensors = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        return {
            "scope": scope,
            "changed": True,
            "enabled_modules": enabled_modules,
            "trainable_params": sum(param.numel() for _, param in trainable_tensors),
            "trainable_tensors": len(trainable_tensors),
            "sample_trainable_names": [name for name, _ in trainable_tensors[:40]],
        }

    def enable_residual_fusion(name: str, module: nn.Module | None) -> None:
        if module is None:
            return
        enabled_any = False
        for attr_name in ("input_norm", "residual"):
            submodule = getattr(module, attr_name, None)
            if isinstance(submodule, nn.Module):
                for param in submodule.parameters():
                    param.requires_grad = True
                enabled_any = True
        gate = getattr(module, "gate", None)
        if isinstance(gate, nn.Parameter):
            gate.requires_grad = True
            enabled_any = True
        if enabled_any:
            enabled_modules.append(name)

    if scope == "action_fusion_residual_only":
        enable_residual_fusion("left_action_token_fusion.residual", getattr(model, "left_action_token_fusion", None))
        enable_residual_fusion("right_action_token_fusion.residual", getattr(model, "right_action_token_fusion", None))
    else:
        enable_module("left_action_token_fusion", getattr(model, "left_action_token_fusion", None))
        enable_module("right_action_token_fusion", getattr(model, "right_action_token_fusion", None))

    if scope in ("action_fusion_head_boundary", "action_fusion_full_action_head"):
        for side_name, agent in (("left", model.left.model), ("right", model.right.model)):
            head = agent.action_head
            for module_name in ("combine",):
                enable_module(f"{side_name}_action_head.{module_name}", getattr(head, module_name, None))
            net = getattr(head, "net", None)
            if net is not None:
                for module_name in ("z_embedder", "final_layer"):
                    enable_module(f"{side_name}_action_head.net.{module_name}", getattr(net, module_name, None))

    if scope == "action_fusion_full_action_head":
        enable_module("left_action_head", model.left.model.action_head)
        enable_module("right_action_head", model.right.model.action_head)

    trainable_tensors = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    return {
        "scope": scope,
        "changed": True,
        "enabled_modules": enabled_modules,
        "trainable_params": sum(param.numel() for _, param in trainable_tensors),
        "trainable_tensors": len(trainable_tensors),
        "sample_trainable_names": [name for name, _ in trainable_tensors[:40]],
    }


def cosine_floor_schedule(optimizer: AdamW, *, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> LambdaLR:
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps + 1)
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    from commvla.models.dual_arm import CommVLANativeV3Config, CommVLANativeV3Pair

    args = parse_args()
    if args.stop_after_step is not None and not (0 < args.stop_after_step <= args.max_steps):
        raise ValueError("--stop-after-step must be in [1, max_steps]")
    if args.resume_from_checkpoint and args.native_checkpoint:
        raise ValueError("Use only one of --resume-from-checkpoint and --native-checkpoint")
    if args.interaction_curriculum == "none" and args.curriculum_transition_step is not None:
        raise ValueError("--curriculum-transition-step requires an interaction curriculum")
    if args.interaction_curriculum != "none":
        if args.curriculum_transition_step is None:
            raise ValueError("--curriculum-transition-step is required for a staged curriculum")
        if not 0 < args.curriculum_transition_step < args.max_steps:
            raise ValueError("--curriculum-transition-step must be in (0, max_steps)")
    project_root = Path(args.project_root)
    add_project_paths(project_root)

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    from twinvla.datasets import load_datasets
    from twinvla.datasets.rlds.utils.data_utils import save_dataset_statistics

    rank, world_size, local_rank, device = init_distributed(args)
    set_seed(args.seed + rank)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    initialization_checkpoint = args.resume_from_checkpoint or args.native_checkpoint
    if initialization_checkpoint:
        config_overrides = {
            "remote_kv_mode": args.remote_kv_mode,
            "common_kv_mode": args.common_kv_mode,
            "common_fusion_mode": args.common_fusion_mode,
            "action_token_fusion_mode": args.action_token_fusion_mode,
            "decoder_mode": args.decoder_mode,
            "action_transformer_layers": args.action_transformer_layers,
            "action_transformer_heads": args.action_transformer_heads,
            "action_transformer_mlp_ratio": args.action_transformer_mlp_ratio,
        }
        pair = CommVLANativeV3Pair.from_pretrained(
            initialization_checkpoint,
            device=device,
            right_device=args.right_device,
            dtype=dtype,
            config_overrides=config_overrides,
        )
        pair.native_config.loss_reduction = args.loss_reduction
        pair.native_config.remote_kv_scale = args.remote_kv_scale
        pair.native_config.remote_kv_dropout_prob = args.remote_kv_dropout_prob
        if rank == 0:
            initialization_kind = "resume_checkpoint" if args.resume_from_checkpoint else "native_checkpoint"
            print(f"[commvla-native-v3] initialized from {initialization_kind}={initialization_checkpoint}", flush=True)
    else:
        native_cfg = CommVLANativeV3Config(
            singlevla_pretrained_path=args.singlevla_pretrained_path,
            left_agent_path=args.left_agent_path,
            right_agent_path=args.right_agent_path,
            remote_kv_mode=args.remote_kv_mode,
            common_strategy=args.common_strategy,
            common_kv_mode=args.common_kv_mode,
            common_fusion_mode=args.common_fusion_mode,
            action_token_fusion_mode=args.action_token_fusion_mode,
            action_transformer_layers=args.action_transformer_layers,
            action_transformer_heads=args.action_transformer_heads,
            action_transformer_mlp_ratio=args.action_transformer_mlp_ratio,
            decoder_mode=args.decoder_mode,
            loss_reduction=args.loss_reduction,
            remote_kv_scale=args.remote_kv_scale,
            remote_kv_dropout_prob=args.remote_kv_dropout_prob,
            freeze_vision_backbone=args.freeze_vision_backbone,
            freeze_llm_backbone=args.freeze_llm_backbone,
            train_llm_last_n_layers=args.train_llm_last_n_layers,
        )
        pair = CommVLANativeV3Pair(native_cfg, device=device, right_device=args.right_device, dtype=dtype)

    scope_summary = apply_train_scope(pair, args.train_scope)
    if rank == 0:
        print("[commvla-native-v3] train_scope " + json.dumps(scope_summary), flush=True)
    model_args, training_args = build_training_arg_shims(args)
    dataloader, dataset_statistics = load_datasets(pair, model_args, training_args, single_arm=False)
    if rank == 0:
        save_dataset_statistics(dataset_statistics, output_dir)
        print("[commvla-native-v3] dataset loaded", flush=True)

    if world_size > 1:
        if rank == 0:
            print(
                "[commvla-native-v3] wrapping DDP "
                + json.dumps(
                    {
                        "world_size": world_size,
                        "find_unused_parameters": args.ddp_find_unused_parameters,
                        "broadcast_buffers": False,
                    }
                ),
                flush=True,
            )
        pair = DDP(
            pair,
            device_ids=[local_rank],
            find_unused_parameters=args.ddp_find_unused_parameters,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        if rank == 0:
            print("[commvla-native-v3] DDP wrapped", flush=True)

    trainable = sum(p.numel() for p in pair.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pair.parameters())
    if rank == 0:
        print(f"[commvla-native-v3] total_params={total/1e9:.4f}B trainable_params={trainable/1e9:.4f}B", flush=True)

    param_groups, param_group_summary = trainable_parameter_groups(
        pair,
        args.learning_rate,
        args.weight_decay,
        vision_learning_rate=args.vision_learning_rate,
        llm_learning_rate=args.llm_learning_rate,
        head_learning_rate=args.head_learning_rate,
        action_transformer_learning_rate=args.action_transformer_learning_rate,
        action_decoder_learning_rate=args.action_decoder_learning_rate,
    )
    if rank == 0:
        print("[commvla-native-v3] parameter_groups " + json.dumps(param_group_summary), flush=True)
    optimizer = AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(args.max_steps * args.warmup_ratio)
    if args.lr_scheduler_type == "cosine":
        if args.min_lr_ratio > 0:
            scheduler = cosine_floor_schedule(
                optimizer,
                warmup_steps=warmup_steps,
                max_steps=args.max_steps,
                min_lr_ratio=args.min_lr_ratio,
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)
    elif args.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)
    else:
        scheduler = None

    resume_step = 0
    if args.resume_from_checkpoint:
        resume_path = Path(args.resume_from_checkpoint)
        complete_path = resume_path / "TRAINING_STATE_COMPLETE.json"
        state_path = resume_path / "training_states.pth"
        rng_path = resume_path / f"rng_state_rank{rank}.pth"
        if not complete_path.exists():
            raise FileNotFoundError(f"Resume checkpoint is incomplete: missing {complete_path}")
        if not state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint is missing {state_path}")
        if not rng_path.exists():
            raise FileNotFoundError(f"Resume checkpoint is missing rank RNG state: {rng_path}")

        training_state = torch.load(state_path, map_location="cpu")
        saved_args = training_state.get("training_args", {})
        critical_fields = (
            "data_mix",
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "lr_scheduler_type",
            "learning_rate",
            "remote_kv_mode",
            "common_kv_mode",
            "common_fusion_mode",
            "action_token_fusion_mode",
            "decoder_mode",
            "interaction_curriculum",
            "curriculum_transition_step",
            "curriculum_stage1_common_kv_mode",
        )
        mismatches = {
            field: {"checkpoint": saved_args.get(field), "current": getattr(args, field)}
            for field in critical_fields
            if field in saved_args and saved_args.get(field) != getattr(args, field)
        }
        saved_world_size = training_state.get("world_size")
        if saved_world_size is not None and saved_world_size != world_size:
            mismatches["world_size"] = {"checkpoint": saved_world_size, "current": world_size}
        if mismatches:
            raise ValueError("Resume configuration mismatch: " + json.dumps(mismatches, sort_keys=True))

        optimizer.load_state_dict(training_state["optimizer"])
        saved_scheduler = training_state.get("scheduler")
        if scheduler is not None:
            if saved_scheduler is None:
                raise ValueError("Resume checkpoint does not contain scheduler state")
            scheduler.load_state_dict(saved_scheduler)
        elif saved_scheduler is not None:
            raise ValueError("Resume checkpoint contains scheduler state but current run has no scheduler")
        resume_step = int(training_state["step"])
        if resume_step >= args.max_steps:
            raise ValueError(f"Resume step {resume_step} must be below max_steps {args.max_steps}")
        rng_state = torch.load(rng_path, map_location="cpu")
        restore_rng_state(rng_state, device)
        if rank == 0:
            print(
                "[commvla-native-v3] resumed "
                + json.dumps(
                    {
                        "checkpoint": str(resume_path),
                        "step": resume_step,
                        "scheduler_last_epoch": scheduler.last_epoch if scheduler is not None else None,
                        "world_size": world_size,
                    }
                ),
                flush=True,
            )

    pair.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    last_log = time.time()
    jsonl_path = output_dir / "training_log.jsonl"

    save_at_steps = None
    if args.save_at_steps:
        save_at_steps = {int(item.strip()) for item in args.save_at_steps.split(",") if item.strip()}
        invalid_save_steps = sorted(step for step in save_at_steps if step <= 0 or step > args.max_steps)
        if invalid_save_steps:
            raise ValueError(f"Invalid --save-at-steps entries for max_steps={args.max_steps}: {invalid_save_steps}")
        save_at_steps.add(args.max_steps)
        if rank == 0:
            print(f"[commvla-native-v3] explicit_save_steps={sorted(save_at_steps)}", flush=True)

    step = resume_step
    stop_reason = None
    active_curriculum_phase = None
    validated_communication_phases: set[str] = set()
    for batch_idx, batch in enumerate(dataloader):
        module = pair.module if isinstance(pair, DDP) else pair
        next_optimizer_step = step + 1
        curriculum_phase = curriculum_phase_for_step(args, next_optimizer_step)
        if curriculum_phase != active_curriculum_phase:
            phase_profile = apply_curriculum_phase(module, args, curriculum_phase)
            module.reset_communication_profile()
            active_curriculum_phase = curriculum_phase
            if rank == 0:
                print(
                    "[commvla-native-v3] curriculum_phase "
                    + json.dumps(
                        {
                            **phase_profile,
                            "completed_optimizer_step": step,
                            "next_optimizer_step": next_optimizer_step,
                            "transition_step": args.curriculum_transition_step,
                            "scheduler_last_epoch": scheduler.last_epoch if scheduler is not None else None,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            outputs = pair(batch)
            loss = outputs["loss"] / args.gradient_accumulation_steps
        if batch_idx == 0 and (
            args.require_zero_model_communication or args.require_private_only_model_communication
        ):
            module = pair.module if isinstance(pair, DDP) else pair
            communication_profile = module.communication_profile()
            if args.require_zero_model_communication:
                if communication_profile["total"] != 0:
                    raise RuntimeError(
                        "Expected zero model communication, got "
                        + json.dumps(communication_profile, sort_keys=True)
                    )
                if rank == 0:
                    print(
                        "[commvla-native-v3] zero_model_communication "
                        + json.dumps(communication_profile, sort_keys=True),
                        flush=True,
                    )
            if args.require_private_only_model_communication:
                is_private_only = (
                    communication_profile["private_kv"] > 0
                    and communication_profile["common"] == 0
                    and communication_profile["action_intent"] == 0
                    and communication_profile["total"] == communication_profile["private_kv"]
                )
                if not is_private_only:
                    raise RuntimeError(
                        "Expected private-only model communication, got "
                        + json.dumps(communication_profile, sort_keys=True)
                    )
                if rank == 0:
                    print(
                        "[commvla-native-v3] private_only_model_communication "
                        + json.dumps(communication_profile, sort_keys=True),
                        flush=True,
                    )
        if (
            args.require_expected_model_communication
            or args.require_staged_model_communication
        ) and curriculum_phase not in validated_communication_phases:
            communication_profile = module.communication_profile()
            expected = expected_communication_groups(args, curriculum_phase)
            validate_communication_groups(
                communication_profile,
                expected,
                label=curriculum_phase,
            )
            validated_communication_phases.add(curriculum_phase)
            if rank == 0:
                print(
                    "[commvla-native-v3] expected_model_communication "
                    + json.dumps(
                        {
                            "phase": curriculum_phase,
                            "expected_nonzero": expected,
                            "profile": communication_profile,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        loss.backward()

        if (batch_idx + 1) % args.gradient_accumulation_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(pair.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if rank == 0 and (step == 1 or step % args.log_steps == 0):
            log = {
                "step": step,
                "loss": float(outputs["loss"].detach().cpu()),
                "loss_left": float(outputs["loss_left"].detach().cpu()),
                "loss_right": float(outputs["loss_right"].detach().cpu()),
                "curriculum_phase": curriculum_phase,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "param_group_lrs": {group.get("name", str(idx)): group["lr"] for idx, group in enumerate(optimizer.param_groups)},
                "seconds": time.time() - started,
                "seconds_since_last_log": time.time() - last_log,
            }
            last_log = time.time()
            print("[commvla-native-v3] " + json.dumps(log), flush=True)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")

        request_stop = bool(args.checkpoint_request_file and Path(args.checkpoint_request_file).exists())
        reached_stop_after = args.stop_after_step is not None and step >= args.stop_after_step
        should_save = request_stop or (reached_stop_after and not args.skip_save_on_stop_after) or (step > 0 and (
            step in save_at_steps if save_at_steps is not None else step % args.save_steps == 0
        ))
        if should_save:
            save_path = output_dir if step == args.max_steps else Path(f"{args.output_dir}-{step}")
            if rank == 0:
                print(f"[commvla-native-v3] saving checkpoint step={step} path={save_path}", flush=True)
                module = pair.module if isinstance(pair, DDP) else pair
                module.save_pretrained(save_path)
                save_dataset_statistics(dataset_statistics, save_path)
                stats_path = save_path / "dataset_statistics.json"
                if stats_path.exists():
                    shutil.copy2(stats_path, save_path / "left_private_agent" / "dataset_statistics.json")
                    shutil.copy2(stats_path, save_path / "right_private_agent" / "dataset_statistics.json")
                atomic_torch_save(
                    {
                        "format_version": 2,
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "world_size": world_size,
                        "training_args": vars(args),
                    },
                    save_path / "training_states.pth",
                )
            if world_size > 1:
                dist.barrier()
            atomic_torch_save(capture_rng_state(device), save_path / f"rng_state_rank{rank}.pth")
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                completion = {
                    "format_version": 2,
                    "step": step,
                    "world_size": world_size,
                    "scheduler_last_epoch": scheduler.last_epoch if scheduler is not None else None,
                    "checkpoint_request": request_stop,
                    "stopped_after_step": reached_stop_after,
                    "interaction_curriculum": args.interaction_curriculum,
                    "curriculum_transition_step": args.curriculum_transition_step,
                    "curriculum_phase": curriculum_phase_for_step(args, step),
                }
                (save_path / "TRAINING_STATE_COMPLETE.json").write_text(
                    json.dumps(completion, indent=2), encoding="utf-8"
                )
                if request_stop:
                    Path(args.checkpoint_request_file).unlink(missing_ok=True)
            if world_size > 1:
                dist.barrier()

        if request_stop:
            stop_reason = "checkpoint_request"
            break
        if reached_stop_after:
            stop_reason = "stop_after_step"
            break

        if step >= args.max_steps:
            stop_reason = "max_steps"
            break

    if rank == 0:
        status = {
            "finished": step >= args.max_steps,
            "step": step,
            "max_steps": args.max_steps,
            "stop_reason": stop_reason,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "interaction_curriculum": args.interaction_curriculum,
            "curriculum_transition_step": args.curriculum_transition_step,
            "curriculum_phase": curriculum_phase_for_step(args, step),
            "seconds": time.time() - started,
            "output_dir": str(output_dir),
        }
        (output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print("[commvla-native-v3] finished " + json.dumps(status), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
