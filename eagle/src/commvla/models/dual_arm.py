"""CommVLA-native v3 model.

The v3 architecture keeps two independent SingleVLA agents.  Each agent builds
its own local common workspace from the common inputs, keeps its own private arm
tokens, and receives the peer agent's private K/V through an explicit Remote-KV
interface.  The default decoder path also keeps independent left/right action
heads so the trained checkpoint can be split into two deployable agents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from twinvla.model.singlevla import SingleVLA

from .joint_action_transformer import JointActionTokenTransformer

Side = Literal["left", "right"]
RemoteKVMode = Literal["full", "no_remote", "random_remote", "stale_remote"]
CommonStrategy = Literal["local_common"]
CommonKVMode = Literal["local", "left_shared", "right_shared", "avg_shared", "sync_qkv_avg_mlp_avg"]
DecoderMode = Literal["separate_decoders", "shared_decoder_batched"]
CommonFusionMode = Literal["fixed_avg", "layer_gate"]
ActionTokenFusionMode = Literal[
    "none", "residual_exchange", "paired_concat_project", "paired_residual_mlp", "paired_joint_transformer"
]
EXTRA_STATE_FILE = "commvla_native_v3_extra.pt"


@dataclass
class CommVLANativeV3Config:
    singlevla_pretrained_path: str
    left_agent_path: str | None = None
    right_agent_path: str | None = None
    remote_kv_mode: RemoteKVMode = "full"
    common_strategy: CommonStrategy = "local_common"
    common_kv_mode: CommonKVMode = "local"
    common_fusion_mode: CommonFusionMode = "fixed_avg"
    action_token_fusion_mode: ActionTokenFusionMode = "none"
    action_transformer_layers: int = 3
    action_transformer_heads: int = 8
    action_transformer_mlp_ratio: int = 2
    decoder_mode: DecoderMode = "separate_decoders"
    loss_reduction: str = "sum"
    remote_kv_scale: float = 1.0
    remote_kv_dropout_prob: float = 0.0
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    train_llm_last_n_layers: int = 0


class ResidualMLPActionTokenFusion(nn.Module):
    """M1-style linear fusion with a zero-init nonlinear residual branch."""

    def __init__(self, hidden_size: int, *, dtype: torch.dtype, device: str | int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.base = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size * 4)
        self.residual = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.gate = nn.Parameter(torch.zeros((), dtype=dtype))
        self.to(device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.base.weight)
        nn.init.zeros_(self.base.bias)
        with torch.no_grad():
            eye = torch.eye(self.hidden_size, device=self.base.weight.device, dtype=self.base.weight.dtype)
            self.base.weight[:, : self.hidden_size].copy_(eye)
        final = self.residual[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            self.gate.fill_(0.05)

    def load_base_linear_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing = {"weight", "bias"} - set(state_dict)
        if missing:
            raise KeyError(f"Missing base linear keys: {sorted(missing)}")
        self.base.load_state_dict({"weight": state_dict["weight"], "bias": state_dict["bias"]})

    def forward(self, pair_token: torch.Tensor) -> torch.Tensor:
        local, remote = pair_token.split(self.hidden_size, dim=-1)
        residual_input = torch.cat([local, remote, local - remote, local * remote], dim=-1)
        residual = self.residual(self.input_norm(residual_input))
        return self.base(pair_token) + self.gate.to(dtype=residual.dtype) * residual


class CommVLANativeV3Pair(nn.Module):
    """Two independent SingleVLA agents with private Remote-KV exchange."""

    def __init__(
        self,
        config: CommVLANativeV3Config,
        *,
        device: str | int = "cuda",
        right_device: str | int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.native_config = config
        self.left_device = device
        self.right_device = right_device if right_device is not None else device
        self.dtype = dtype

        left_path = config.left_agent_path or config.singlevla_pretrained_path
        right_path = config.right_agent_path or config.singlevla_pretrained_path
        self.left = SingleVLA(pretrained_path=left_path, device=self.left_device, dtype=dtype)
        self.right = SingleVLA(pretrained_path=right_path, device=self.right_device, dtype=dtype)
        self.left.model.config.use_cache = False
        self.right.model.config.use_cache = False

        self.config = SimpleNamespace(
            normalization=self.left.config.normalization,
            global_normalization=getattr(self.left.config, "global_normalization", False),
            action_head=self.left.config.action_head,
            action_len=self.left.config.action_len,
            action_dim=self.left.config.action_dim,
            state_dim=self.left.config.state_dim,
        )
        self.hidden_size = int(self.left.model.hidden_dim())
        self.state_dim = int(self.left.config.state_dim)
        self.action_len = int(self.left.config.action_len)
        self.action_dim = int(self.left.config.action_dim)
        self._runtime_common_kv_mode: str | None = None
        self._runtime_private_remote_kv_mode: str | None = None
        self._runtime_action_token_ablation_mode: str | None = None
        self._communication_totals: dict[str, int] = {"common": 0, "private_kv": 0, "action_intent": 0}

        if config.common_strategy != "local_common":
            raise ValueError(f"Unsupported common_strategy for v3: {config.common_strategy}")
        if config.common_kv_mode not in ("local", "left_shared", "right_shared", "avg_shared", "sync_qkv_avg_mlp_avg"):
            raise ValueError(f"Unsupported common_kv_mode for v3: {config.common_kv_mode}")
        if config.common_fusion_mode not in ("fixed_avg", "layer_gate"):
            raise ValueError(f"Unsupported common_fusion_mode for v3: {config.common_fusion_mode}")
        if config.action_token_fusion_mode not in (
            "none", "residual_exchange", "paired_concat_project", "paired_residual_mlp", "paired_joint_transformer"
        ):
            raise ValueError(f"Unsupported action_token_fusion_mode for v3: {config.action_token_fusion_mode}")
        if config.decoder_mode not in ("separate_decoders", "shared_decoder_batched"):
            raise ValueError(f"Unsupported decoder_mode for v3: {config.decoder_mode}")
        if config.remote_kv_scale < 0:
            raise ValueError(f"remote_kv_scale must be non-negative, got {config.remote_kv_scale}")
        if not 0.0 <= config.remote_kv_dropout_prob <= 1.0:
            raise ValueError(f"remote_kv_dropout_prob must be in [0, 1], got {config.remote_kv_dropout_prob}")

        self.shared_action_head = self.left.model.action_head if config.decoder_mode == "shared_decoder_batched" else None
        if config.decoder_mode == "shared_decoder_batched":
            for param in self.right.model.action_head.parameters():
                param.requires_grad = False

        num_layers = len(self.left.model.text_backbone().model.layers)
        self.common_layer_gate: nn.Parameter | None = None
        if config.common_fusion_mode == "layer_gate":
            self.common_layer_gate = nn.Parameter(torch.zeros(num_layers, device=self.left_device, dtype=dtype))

        self.left_action_token_fusion: nn.Module | None = None
        self.right_action_token_fusion: nn.Module | None = None
        self.joint_action_token_transformer: JointActionTokenTransformer | None = None
        if config.action_token_fusion_mode == "paired_joint_transformer":
            self.joint_action_token_transformer = JointActionTokenTransformer(
                self.hidden_size,
                num_layers=config.action_transformer_layers,
                num_heads=config.action_transformer_heads,
                mlp_ratio=config.action_transformer_mlp_ratio,
                device=self.left_device,
                dtype=dtype,
            )
        if config.action_token_fusion_mode in ("residual_exchange", "paired_concat_project", "paired_residual_mlp"):
            if config.action_token_fusion_mode == "paired_residual_mlp":
                self.left_action_token_fusion = ResidualMLPActionTokenFusion(
                    self.hidden_size, device=self.left_device, dtype=dtype
                )
                self.right_action_token_fusion = ResidualMLPActionTokenFusion(
                    self.hidden_size, device=self.right_device, dtype=dtype
                )
            else:
                self.left_action_token_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device=self.left_device, dtype=dtype)
                self.right_action_token_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device=self.right_device, dtype=dtype)
                nn.init.zeros_(self.left_action_token_fusion.weight)
                nn.init.zeros_(self.left_action_token_fusion.bias)
                nn.init.zeros_(self.right_action_token_fusion.weight)
                nn.init.zeros_(self.right_action_token_fusion.bias)
                if config.action_token_fusion_mode == "paired_concat_project":
                    with torch.no_grad():
                        eye_l = torch.eye(self.hidden_size, device=self.left_device, dtype=dtype)
                        eye_r = torch.eye(self.hidden_size, device=self.right_device, dtype=dtype)
                        self.left_action_token_fusion.weight[:, : self.hidden_size].copy_(eye_l)
                        self.right_action_token_fusion.weight[:, : self.hidden_size].copy_(eye_r)

        if config.freeze_vision_backbone:
            self.freeze_vision_backbone()
        if config.freeze_llm_backbone:
            self.freeze_llm_backbone()

    def freeze_vision_backbone(self) -> None:
        for agent in (self.left.model, self.right.model):
            for param in agent.vision_backbone().parameters():
                param.requires_grad = False

    def freeze_llm_backbone(self) -> None:
        for agent in (self.left.model, self.right.model):
            for param in agent.text_backbone().parameters():
                param.requires_grad = False
            if self.native_config.train_llm_last_n_layers > 0:
                layers = agent.text_backbone().model.layers
                for layer in layers[-self.native_config.train_llm_last_n_layers :]:
                    for param in layer.parameters():
                        param.requires_grad = True
                norm = getattr(agent.text_backbone().model, "norm", None)
                if norm is not None:
                    for param in norm.parameters():
                        param.requires_grad = True
            for module_name in ("embed_arm_state", "agg"):
                module = getattr(agent, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
            action_token = getattr(agent, "action_token", None)
            if action_token is not None:
                action_token.requires_grad_(True)

        if self.native_config.decoder_mode == "shared_decoder_batched":
            assert self.shared_action_head is not None
            for param in self.shared_action_head.parameters():
                param.requires_grad = True
        else:
            for agent in (self.left.model, self.right.model):
                for param in agent.action_head.parameters():
                    param.requires_grad = True

    def reset_remote_kv_cache(self) -> None:
        self._stale_remote_cache = None

    def set_runtime_communication_modes(
        self,
        *,
        common_kv_mode: str | None = None,
        private_remote_kv_mode: str | None = None,
        action_token_ablation_mode: str | None = None,
    ) -> None:
        if common_kv_mode is not None and common_kv_mode not in {
            "local", "left_shared", "right_shared", "avg_shared", "sync_qkv_avg_mlp_avg"
        }:
            raise ValueError(f"Unsupported runtime common_kv_mode: {common_kv_mode}")
        if private_remote_kv_mode is not None and private_remote_kv_mode not in {
            "full", "no_remote", "random_remote", "stale_remote"
        }:
            raise ValueError(f"Unsupported runtime private_remote_kv_mode: {private_remote_kv_mode}")
        if action_token_ablation_mode is not None and action_token_ablation_mode not in {
            "none", "random_remote", "disable_fusion"
        }:
            raise ValueError(f"Unsupported runtime action_token_ablation_mode: {action_token_ablation_mode}")
        self._runtime_common_kv_mode = common_kv_mode
        self._runtime_private_remote_kv_mode = private_remote_kv_mode
        self._runtime_action_token_ablation_mode = action_token_ablation_mode

    def clear_runtime_communication_modes(self) -> None:
        self._runtime_common_kv_mode = None
        self._runtime_private_remote_kv_mode = None
        self._runtime_action_token_ablation_mode = None

    def reset_communication_profile(self) -> None:
        self._communication_totals = {"common": 0, "private_kv": 0, "action_intent": 0}

    def communication_profile(self) -> dict[str, int]:
        totals = dict(self._communication_totals)
        totals["total"] = sum(totals.values())
        return totals

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _record_pair_message(self, group: str, left: torch.Tensor, right: torch.Tensor) -> None:
        self._communication_totals[group] += self._tensor_bytes(left) + self._tensor_bytes(right)

    def _effective_common_kv_mode(self) -> str:
        return self._runtime_common_kv_mode or self.native_config.common_kv_mode

    def _effective_private_remote_kv_mode(self) -> str:
        return self._runtime_private_remote_kv_mode or self.native_config.remote_kv_mode

    def preprocess_inputs(self, image, image_wrist_r, image_wrist_l, instruction, action=None):
        """Preprocess a dual-arm sample into left/right SingleVLA input fields."""

        left = self.left.model.preprocess_inputs(image, image_wrist_l, instruction, action=None)
        right = self.right.model.preprocess_inputs(image, image_wrist_r, instruction, action=None)
        output = {}
        for key, value in left.items():
            output[f"{key}_left"] = value
        for key, value in right.items():
            output[f"{key}_right"] = value
        return output

    @staticmethod
    def _to_device(batch: BatchFeature | dict, device: torch.device | str, dtype: torch.dtype) -> BatchFeature:
        out = BatchFeature()
        for key, value in batch.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    out[key] = value.to(device=device, dtype=dtype)
                else:
                    out[key] = value.to(device=device)
            else:
                out[key] = value
        return out

    def _side_batch(self, batch: BatchFeature | dict, side: Side) -> BatchFeature:
        suffix = f"_{side}"
        device = self.left_device if side == "left" else self.right_device
        side_batch = BatchFeature()
        for key in ("ci_ids", "modal_ids", "input_ids", "label_ids", "attention_mask", "pixel_values_primary", "pixel_values_wrist"):
            side_batch[key] = batch[f"{key}{suffix}"]

        if side == "left":
            side_batch["proprio"] = batch["proprio"][..., : self.state_dim]
            if "action" in batch:
                side_batch["action"] = batch["action"][..., : self.action_dim]
        else:
            side_batch["proprio"] = batch["proprio"][..., self.state_dim :]
            if "action" in batch:
                side_batch["action"] = batch["action"][..., self.action_dim :]
        return self._to_device(side_batch, device, self.dtype)

    @staticmethod
    def _select(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz = hidden.shape[0]
        return hidden[mask].reshape(bsz, -1, hidden.shape[-1])

    @staticmethod
    def _select_2d(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz = values.shape[0]
        return values[mask].reshape(bsz, -1)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def _apply_rotary_pos_emb(cls, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (q * cos) + (cls._rotate_half(q) * sin), (k * cos) + (cls._rotate_half(k) * sin)

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _causal_mask(valid: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = valid.shape
        device = valid.device
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        allowed = causal.unsqueeze(0) & valid.bool().unsqueeze(1).expand(-1, seq_len, -1)
        mask = torch.full((bsz, 1, seq_len, seq_len), torch.finfo(torch.float32).min, device=device)
        return mask.masked_fill(allowed.unsqueeze(1), 0.0)

    @staticmethod
    def _private_attention_mask(
        common_valid: torch.Tensor,
        local_valid: torch.Tensor,
        remote_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, common_len = common_valid.shape
        local_len = local_valid.shape[1]
        remote_len = 0 if remote_valid is None else remote_valid.shape[1]
        device = local_valid.device

        common_allowed = common_valid.bool().unsqueeze(1).expand(-1, local_len, -1)
        order = torch.arange(local_len, device=device)
        tri = order[None, :] <= order[:, None]
        local_allowed = tri.unsqueeze(0) & local_valid.bool().unsqueeze(1).expand(-1, local_len, -1)
        pieces = [common_allowed, local_allowed]
        if remote_valid is not None:
            remote_allowed = tri.unsqueeze(0) & remote_valid.bool().unsqueeze(1).expand(-1, local_len, -1)
            pieces.append(remote_allowed)
        allowed = torch.cat(pieces, dim=-1)
        total_len = common_len + local_len + remote_len
        mask = torch.full((bsz, 1, local_len, total_len), torch.finfo(torch.float32).min, device=device)
        return mask.masked_fill(allowed.unsqueeze(1), 0.0)

    def _qkv_with_positions(self, layer, hidden: torch.Tensor, rotary_emb, position_ids: torch.Tensor):
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden)
        q, k, v = self._project_qkv(layer, normed)
        q, k = self._apply_rotary_pos_emb(q, k, *rotary_emb(hidden, position_ids))
        return normed, q, k, v

    @staticmethod
    def _project_qkv(layer, normed: torch.Tensor):
        attn = layer.self_attn
        shape = (*normed.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(normed).view(shape).transpose(1, 2)
        k = attn.k_proj(normed).view(shape).transpose(1, 2)
        v = attn.v_proj(normed).view(shape).transpose(1, 2)
        return q, k, v

    @staticmethod
    def _avg_like_left(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left + right.to(device=left.device, dtype=left.dtype)) / 2

    def _common_fuse_like_left(self, left: torch.Tensor, right: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.native_config.common_fusion_mode != "layer_gate":
            return self._avg_like_left(left, right)
        if self.common_layer_gate is None:
            raise RuntimeError("common_layer_gate is required when common_fusion_mode='layer_gate'")
        gate = torch.sigmoid(self.common_layer_gate[layer_idx]).to(device=left.device, dtype=left.dtype)
        right = right.to(device=left.device, dtype=left.dtype)
        return gate * left + (1.0 - gate) * right

    def _attention_values(self, layer, query, key, value, mask: torch.Tensor) -> torch.Tensor:
        attn = layer.self_attn
        key_states = self._repeat_kv(key, attn.num_key_value_groups)
        value_states = self._repeat_kv(value, attn.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * attn.scaling
        weights = weights + mask.to(dtype=weights.dtype)
        weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output.reshape(output.shape[0], output.shape[1], -1)

    def _attention(self, layer, query, key, value, mask: torch.Tensor) -> torch.Tensor:
        return layer.self_attn.o_proj(self._attention_values(layer, query, key, value, mask))

    @staticmethod
    def _random_like(tensor: torch.Tensor) -> torch.Tensor:
        # Random-message ablations must not alter the diffusion sampler's RNG stream.
        if tensor.is_cuda:
            state = torch.cuda.get_rng_state(tensor.device)
            value = torch.randn_like(tensor)
            torch.cuda.set_rng_state(state, tensor.device)
            return value
        state = torch.random.get_rng_state()
        value = torch.randn_like(tensor)
        torch.random.set_rng_state(state)
        return value

    def _drop_remote_kv_this_forward(self, device: torch.device | str) -> bool:
        if not self.training or self.native_config.remote_kv_dropout_prob <= 0:
            return False
        return bool(torch.rand((), device=device).item() < self.native_config.remote_kv_dropout_prob)

    def _remote_value_scale(self) -> float:
        if not self.training:
            return 1.0
        return float(self.native_config.remote_kv_scale)

    def _three_way_forward(self, left_batch: BatchFeature, right_batch: BatchFeature):
        left_model = self.left.model
        right_model = self.right.model

        hidden_l_full, mask_l_full = left_model.prepare_embeds(left_batch)
        hidden_r_full, mask_r_full = right_model.prepare_embeds(right_batch)

        common_mask_l = left_batch["ci_ids"] == 0
        common_mask_r = right_batch["ci_ids"] == 0
        private_mask_l = left_batch["ci_ids"] == 1
        private_mask_r = right_batch["ci_ids"] == 1

        hidden_c_l = self._select(hidden_l_full, common_mask_l)
        hidden_c_r = self._select(hidden_r_full, common_mask_r)
        hidden_l = self._select(hidden_l_full, private_mask_l)
        hidden_r = self._select(hidden_r_full, private_mask_r)

        common_valid_l = self._select_2d(mask_l_full, common_mask_l).to(device=hidden_c_l.device)
        common_valid_r = self._select_2d(mask_r_full, common_mask_r).to(device=hidden_c_r.device)
        private_valid_l = self._select_2d(mask_l_full, private_mask_l).to(device=hidden_l.device)
        private_valid_r = self._select_2d(mask_r_full, private_mask_r).to(device=hidden_r.device)
        private_modal_l = self._select_2d(left_batch["modal_ids"], private_mask_l).to(device=hidden_l.device)
        private_modal_r = self._select_2d(right_batch["modal_ids"], private_mask_r).to(device=hidden_r.device)

        bsz = hidden_l.shape[0]
        common_len_l = hidden_c_l.shape[1]
        common_len_r = hidden_c_r.shape[1]
        private_len_l = hidden_l.shape[1]
        private_len_r = hidden_r.shape[1]
        pos_c_l = torch.arange(common_len_l, device=hidden_c_l.device).unsqueeze(0).expand(bsz, -1)
        pos_c_r = torch.arange(common_len_r, device=hidden_c_r.device).unsqueeze(0).expand(bsz, -1)
        pos_l = (common_len_l + torch.arange(private_len_l, device=hidden_l.device)).unsqueeze(0).expand(bsz, -1)
        pos_r = (common_len_r + torch.arange(private_len_r, device=hidden_r.device)).unsqueeze(0).expand(bsz, -1)

        layers_c_l = left_model.text_backbone().model.layers
        layers_c_r = right_model.text_backbone().model.layers
        layers_l = left_model.text_backbone().model.layers
        layers_r = right_model.text_backbone().model.layers
        rotary_c_l = left_model.text_backbone().model.rotary_emb
        rotary_c_r = right_model.text_backbone().model.rotary_emb
        rotary_l = left_model.text_backbone().model.rotary_emb
        rotary_r = right_model.text_backbone().model.rotary_emb

        common_attn_mask_l = self._causal_mask(common_valid_l)
        common_attn_mask_r = self._causal_mask(common_valid_r)

        next_stale_cache = []
        stale_cache = getattr(self, "_stale_remote_cache", None)
        drop_remote_kv = self._drop_remote_kv_this_forward(hidden_l.device)
        remote_value_scale = self._remote_value_scale()

        common_mode = self._effective_common_kv_mode()
        private_mode = self._effective_private_remote_kv_mode()

        if common_mode == "sync_qkv_avg_mlp_avg":
            if common_len_l != common_len_r:
                raise ValueError(f"v3.6 common sync requires equal common token counts, got {common_len_l} and {common_len_r}")
            self._record_pair_message("common", hidden_c_l, hidden_c_r)
            self._record_pair_message("common", common_valid_l, common_valid_r)
            hidden_c = self._avg_like_left(hidden_c_l, hidden_c_r)
            common_valid_sync = (common_valid_l.bool() & common_valid_r.to(device=common_valid_l.device).bool()).to(
                dtype=common_valid_l.dtype
            )
            common_attn_mask_sync = self._causal_mask(common_valid_sync)

            for layer_idx, (layer_c_l, layer_c_r, layer_l, layer_r) in enumerate(zip(layers_c_l, layers_c_r, layers_l, layers_r)):
                residual_c = hidden_c
                hidden_c_for_r = hidden_c.to(device=hidden_c_r.device, dtype=hidden_c_r.dtype)
                norm_c_l = layer_c_l.input_layernorm(hidden_c)
                norm_c_r = layer_c_r.input_layernorm(hidden_c_for_r)
                self._record_pair_message("common", norm_c_l, norm_c_r)
                norm_c = self._common_fuse_like_left(norm_c_l, norm_c_r, layer_idx)

                q_c_l, k_c_l, v_c_l = self._project_qkv(layer_c_l, norm_c)
                norm_c_for_r = norm_c.to(device=hidden_c_r.device, dtype=hidden_c_r.dtype)
                q_c_r, k_c_r, v_c_r = self._project_qkv(layer_c_r, norm_c_for_r)
                self._record_pair_message("common", q_c_l, q_c_r)
                self._record_pair_message("common", k_c_l, k_c_r)
                self._record_pair_message("common", v_c_l, v_c_r)
                q_c = self._common_fuse_like_left(q_c_l, q_c_r, layer_idx)
                k_c = self._common_fuse_like_left(k_c_l, k_c_r, layer_idx)
                v_c = self._common_fuse_like_left(v_c_l, v_c_r, layer_idx)
                q_c, k_c = self._apply_rotary_pos_emb(q_c, k_c, *rotary_c_l(hidden_c, pos_c_l))

                common_attn_values = self._attention_values(layer_c_l, q_c, k_c, v_c, common_attn_mask_sync)
                common_attn_l = layer_c_l.self_attn.o_proj(common_attn_values)
                common_attn_r = layer_c_r.self_attn.o_proj(
                    common_attn_values.to(device=hidden_c_r.device, dtype=hidden_c_r.dtype)
                )
                self._record_pair_message("common", common_attn_l, common_attn_r)
                hidden_c = residual_c + self._common_fuse_like_left(common_attn_l, common_attn_r, layer_idx)

                residual_c = hidden_c
                hidden_c_for_r = hidden_c.to(device=hidden_c_r.device, dtype=hidden_c_r.dtype)
                mlp_norm_c_l = layer_c_l.post_attention_layernorm(hidden_c)
                mlp_norm_c_r = layer_c_r.post_attention_layernorm(hidden_c_for_r)
                self._record_pair_message("common", mlp_norm_c_l, mlp_norm_c_r)
                mlp_norm_c = self._common_fuse_like_left(mlp_norm_c_l, mlp_norm_c_r, layer_idx)
                mlp_c_l = layer_c_l.mlp(mlp_norm_c)
                mlp_c_r = layer_c_r.mlp(mlp_norm_c.to(device=hidden_c_r.device, dtype=hidden_c_r.dtype))
                self._record_pair_message("common", mlp_c_l, mlp_c_r)
                hidden_c = residual_c + self._common_fuse_like_left(mlp_c_l, mlp_c_r, layer_idx)

                residual_l = hidden_l
                residual_r = hidden_r
                _, q_l, k_l, v_l = self._qkv_with_positions(layer_l, hidden_l, rotary_l, pos_l)
                _, q_r, k_r, v_r = self._qkv_with_positions(layer_r, hidden_r, rotary_r, pos_r)

                mode = private_mode
                if mode == "full" and not drop_remote_kv:
                    self._record_pair_message("private_kv", k_l, k_r)
                    self._record_pair_message("private_kv", v_l, v_r)
                    self._record_pair_message("private_kv", private_valid_l, private_valid_r)
                    k_remote_for_l = k_r.to(device=k_l.device)
                    v_remote_for_l = v_r.to(device=v_l.device) * remote_value_scale
                    k_remote_for_r = k_l.to(device=k_r.device)
                    v_remote_for_r = v_l.to(device=v_r.device) * remote_value_scale
                    remote_valid_for_l = private_valid_r.to(device=hidden_l.device)
                    remote_valid_for_r = private_valid_l.to(device=hidden_r.device)
                elif mode == "stale_remote":
                    if stale_cache is not None and layer_idx < len(stale_cache):
                        cached = stale_cache[layer_idx]
                        k_remote_for_l = cached["k_r"].to(device=k_l.device, dtype=k_l.dtype)
                        v_remote_for_l = cached["v_r"].to(device=v_l.device, dtype=v_l.dtype)
                        k_remote_for_r = cached["k_l"].to(device=k_r.device, dtype=k_r.dtype)
                        v_remote_for_r = cached["v_l"].to(device=v_r.device, dtype=v_r.dtype)
                        remote_valid_for_l = cached["valid_r"].to(device=hidden_l.device)
                        remote_valid_for_r = cached["valid_l"].to(device=hidden_r.device)
                    else:
                        k_remote_for_l = k_r.to(device=k_l.device)
                        v_remote_for_l = v_r.to(device=v_l.device)
                        k_remote_for_r = k_l.to(device=k_r.device)
                        v_remote_for_r = v_l.to(device=v_r.device)
                        remote_valid_for_l = private_valid_r.to(device=hidden_l.device)
                        remote_valid_for_r = private_valid_l.to(device=hidden_r.device)
                elif mode == "random_remote":
                    self._record_pair_message("private_kv", k_l, k_r)
                    self._record_pair_message("private_kv", v_l, v_r)
                    self._record_pair_message("private_kv", private_valid_l, private_valid_r)
                    k_remote_for_l = self._random_like(k_r).to(device=k_l.device)
                    v_remote_for_l = self._random_like(v_r).to(device=v_l.device)
                    k_remote_for_r = self._random_like(k_l).to(device=k_r.device)
                    v_remote_for_r = self._random_like(v_l).to(device=v_r.device)
                    remote_valid_for_l = private_valid_r.to(device=hidden_l.device)
                    remote_valid_for_r = private_valid_l.to(device=hidden_r.device)
                else:
                    k_remote_for_l = v_remote_for_l = None
                    k_remote_for_r = v_remote_for_r = None
                    remote_valid_for_l = remote_valid_for_r = None

                k_c_for_l = k_c.to(device=k_l.device, dtype=k_l.dtype)
                v_c_for_l = v_c.to(device=v_l.device, dtype=v_l.dtype)
                common_valid_for_l = common_valid_sync.to(device=hidden_l.device)
                if k_remote_for_l is None:
                    k_for_l = torch.cat([k_c_for_l, k_l], dim=2)
                    v_for_l = torch.cat([v_c_for_l, v_l], dim=2)
                else:
                    k_for_l = torch.cat([k_c_for_l, k_l, k_remote_for_l], dim=2)
                    v_for_l = torch.cat([v_c_for_l, v_l, v_remote_for_l], dim=2)
                mask_for_l = self._private_attention_mask(common_valid_for_l, private_valid_l, remote_valid_for_l)

                k_c_for_r = k_c.to(device=k_r.device, dtype=k_r.dtype)
                v_c_for_r = v_c.to(device=v_r.device, dtype=v_r.dtype)
                common_valid_for_r = common_valid_sync.to(device=hidden_r.device)
                if k_remote_for_r is None:
                    k_for_r = torch.cat([k_c_for_r, k_r], dim=2)
                    v_for_r = torch.cat([v_c_for_r, v_r], dim=2)
                else:
                    k_for_r = torch.cat([k_c_for_r, k_r, k_remote_for_r], dim=2)
                    v_for_r = torch.cat([v_c_for_r, v_r, v_remote_for_r], dim=2)
                mask_for_r = self._private_attention_mask(common_valid_for_r, private_valid_r, remote_valid_for_r)

                hidden_l = residual_l + self._attention(layer_l, q_l, k_for_l, v_for_l, mask_for_l)
                hidden_r = residual_r + self._attention(layer_r, q_r, k_for_r, v_for_r, mask_for_r)
                hidden_l = hidden_l + layer_l.mlp(layer_l.post_attention_layernorm(hidden_l))
                hidden_r = hidden_r + layer_r.mlp(layer_r.post_attention_layernorm(hidden_r))
                if mode == "stale_remote":
                    next_stale_cache.append(
                        {
                            "k_l": k_l.detach(),
                            "v_l": v_l.detach(),
                            "valid_l": private_valid_l.detach(),
                            "k_r": k_r.detach(),
                            "v_r": v_r.detach(),
                            "valid_r": private_valid_r.detach(),
                        }
                    )

            hidden_c_l = left_model.text_backbone().model.norm(hidden_c)
            hidden_c_r = right_model.text_backbone().model.norm(hidden_c.to(device=hidden_c_r.device, dtype=hidden_c_r.dtype))
            hidden_l = left_model.text_backbone().model.norm(hidden_l)
            hidden_r = right_model.text_backbone().model.norm(hidden_r)
            if private_mode == "stale_remote":
                self._stale_remote_cache = next_stale_cache
            return {
                "left_common": hidden_c_l,
                "right_common": hidden_c_r,
                "left_private": hidden_l,
                "right_private": hidden_r,
                "left_private_modal_ids": private_modal_l,
                "right_private_modal_ids": private_modal_r,
            }

        for layer_idx, (layer_c_l, layer_c_r, layer_l, layer_r) in enumerate(zip(layers_c_l, layers_c_r, layers_l, layers_r)):
            residual_c_l = hidden_c_l
            _, q_c_l, k_c_l, v_c_l = self._qkv_with_positions(layer_c_l, hidden_c_l, rotary_c_l, pos_c_l)
            hidden_c_l = residual_c_l + self._attention(layer_c_l, q_c_l, k_c_l, v_c_l, common_attn_mask_l)
            hidden_c_l = hidden_c_l + layer_c_l.mlp(layer_c_l.post_attention_layernorm(hidden_c_l))

            residual_c_r = hidden_c_r
            _, q_c_r, k_c_r, v_c_r = self._qkv_with_positions(layer_c_r, hidden_c_r, rotary_c_r, pos_c_r)
            hidden_c_r = residual_c_r + self._attention(layer_c_r, q_c_r, k_c_r, v_c_r, common_attn_mask_r)
            hidden_c_r = hidden_c_r + layer_c_r.mlp(layer_c_r.post_attention_layernorm(hidden_c_r))

            residual_l = hidden_l
            residual_r = hidden_r
            _, q_l, k_l, v_l = self._qkv_with_positions(layer_l, hidden_l, rotary_l, pos_l)
            _, q_r, k_r, v_r = self._qkv_with_positions(layer_r, hidden_r, rotary_r, pos_r)

            mode = private_mode
            if mode == "full" and not drop_remote_kv:
                self._record_pair_message("private_kv", k_l, k_r)
                self._record_pair_message("private_kv", v_l, v_r)
                self._record_pair_message("private_kv", private_valid_l, private_valid_r)
                k_remote_for_l = k_r.to(device=k_l.device)
                v_remote_for_l = v_r.to(device=v_l.device) * remote_value_scale
                k_remote_for_r = k_l.to(device=k_r.device)
                v_remote_for_r = v_l.to(device=v_r.device) * remote_value_scale
                remote_valid_for_l = private_valid_r.to(device=hidden_l.device)
                remote_valid_for_r = private_valid_l.to(device=hidden_r.device)
            elif mode == "stale_remote":
                if stale_cache is not None and layer_idx < len(stale_cache):
                    cached = stale_cache[layer_idx]
                    k_remote_for_l = cached["k_r"].to(device=k_l.device, dtype=k_l.dtype)
                    v_remote_for_l = cached["v_r"].to(device=v_l.device, dtype=v_l.dtype)
                    k_remote_for_r = cached["k_l"].to(device=k_r.device, dtype=k_r.dtype)
                    v_remote_for_r = cached["v_l"].to(device=v_r.device, dtype=v_r.dtype)
                    remote_valid_for_l = cached["valid_r"].to(device=hidden_l.device)
                    remote_valid_for_r = cached["valid_l"].to(device=hidden_r.device)
                else:
                    k_remote_for_l = k_r.to(device=k_l.device)
                    v_remote_for_l = v_r.to(device=v_l.device)
                    k_remote_for_r = k_l.to(device=k_r.device)
                    v_remote_for_r = v_l.to(device=v_r.device)
                    remote_valid_for_l = private_valid_r.to(device=hidden_l.device)
                    remote_valid_for_r = private_valid_l.to(device=hidden_r.device)
            elif mode == "random_remote":
                self._record_pair_message("private_kv", k_l, k_r)
                self._record_pair_message("private_kv", v_l, v_r)
                self._record_pair_message("private_kv", private_valid_l, private_valid_r)
                k_remote_for_l = self._random_like(k_r).to(device=k_l.device)
                v_remote_for_l = self._random_like(v_r).to(device=v_l.device)
                k_remote_for_r = self._random_like(k_l).to(device=k_r.device)
                v_remote_for_r = self._random_like(v_l).to(device=v_r.device)
                remote_valid_for_l = private_valid_r.to(device=hidden_l.device)
                remote_valid_for_r = private_valid_l.to(device=hidden_r.device)
            else:
                k_remote_for_l = v_remote_for_l = None
                k_remote_for_r = v_remote_for_r = None
                remote_valid_for_l = remote_valid_for_r = None

            if common_mode == "local":
                k_common_for_l = k_c_l
                v_common_for_l = v_c_l
                k_common_for_r = k_c_r
                v_common_for_r = v_c_r
            elif common_mode == "left_shared":
                self._communication_totals["common"] += self._tensor_bytes(k_c_l) + self._tensor_bytes(v_c_l)
                k_common_for_l = k_c_l
                v_common_for_l = v_c_l
                k_common_for_r = k_c_l
                v_common_for_r = v_c_l
            elif common_mode == "right_shared":
                self._communication_totals["common"] += self._tensor_bytes(k_c_r) + self._tensor_bytes(v_c_r)
                k_common_for_l = k_c_r
                v_common_for_l = v_c_r
                k_common_for_r = k_c_r
                v_common_for_r = v_c_r
            else:
                self._record_pair_message("common", k_c_l, k_c_r)
                self._record_pair_message("common", v_c_l, v_c_r)
                k_avg = (k_c_l + k_c_r.to(device=k_c_l.device, dtype=k_c_l.dtype)) / 2
                v_avg = (v_c_l + v_c_r.to(device=v_c_l.device, dtype=v_c_l.dtype)) / 2
                k_common_for_l = k_common_for_r = k_avg
                v_common_for_l = v_common_for_r = v_avg

            k_c_l_for_l = k_common_for_l.to(device=k_l.device, dtype=k_l.dtype)
            v_c_l_for_l = v_common_for_l.to(device=v_l.device, dtype=v_l.dtype)
            common_valid_for_l = common_valid_l.to(device=hidden_l.device)
            if k_remote_for_l is None:
                k_for_l = torch.cat([k_c_l_for_l, k_l], dim=2)
                v_for_l = torch.cat([v_c_l_for_l, v_l], dim=2)
            else:
                k_for_l = torch.cat([k_c_l_for_l, k_l, k_remote_for_l], dim=2)
                v_for_l = torch.cat([v_c_l_for_l, v_l, v_remote_for_l], dim=2)
            mask_for_l = self._private_attention_mask(common_valid_for_l, private_valid_l, remote_valid_for_l)

            k_c_r_for_r = k_common_for_r.to(device=k_r.device, dtype=k_r.dtype)
            v_c_r_for_r = v_common_for_r.to(device=v_r.device, dtype=v_r.dtype)
            common_valid_for_r = common_valid_r.to(device=hidden_r.device)
            if k_remote_for_r is None:
                k_for_r = torch.cat([k_c_r_for_r, k_r], dim=2)
                v_for_r = torch.cat([v_c_r_for_r, v_r], dim=2)
            else:
                k_for_r = torch.cat([k_c_r_for_r, k_r, k_remote_for_r], dim=2)
                v_for_r = torch.cat([v_c_r_for_r, v_r, v_remote_for_r], dim=2)
            mask_for_r = self._private_attention_mask(common_valid_for_r, private_valid_r, remote_valid_for_r)

            hidden_l = residual_l + self._attention(layer_l, q_l, k_for_l, v_for_l, mask_for_l)
            hidden_r = residual_r + self._attention(layer_r, q_r, k_for_r, v_for_r, mask_for_r)
            hidden_l = hidden_l + layer_l.mlp(layer_l.post_attention_layernorm(hidden_l))
            hidden_r = hidden_r + layer_r.mlp(layer_r.post_attention_layernorm(hidden_r))
            if mode == "stale_remote":
                next_stale_cache.append(
                    {
                        "k_l": k_l.detach(),
                        "v_l": v_l.detach(),
                        "valid_l": private_valid_l.detach(),
                        "k_r": k_r.detach(),
                        "v_r": v_r.detach(),
                        "valid_r": private_valid_r.detach(),
                    }
                )

        hidden_c_l = left_model.text_backbone().model.norm(hidden_c_l)
        hidden_c_r = right_model.text_backbone().model.norm(hidden_c_r)
        hidden_l = left_model.text_backbone().model.norm(hidden_l)
        hidden_r = right_model.text_backbone().model.norm(hidden_r)
        if private_mode == "stale_remote":
            self._stale_remote_cache = next_stale_cache
        return {
            "left_common": hidden_c_l,
            "right_common": hidden_c_r,
            "left_private": hidden_l,
            "right_private": hidden_r,
            "left_private_modal_ids": private_modal_l,
            "right_private_modal_ids": private_modal_r,
        }

    def _action_token_from_private(self, agent_model, hidden_private: torch.Tensor, private_modal_ids: torch.Tensor) -> torch.Tensor:
        bsz = private_modal_ids.shape[0]
        action_token = hidden_private[private_modal_ids == 5].reshape(bsz, -1, agent_model.hidden_dim())
        return agent_model.agg(action_token)

    def _fuse_action_tokens(self, left_token: torch.Tensor, right_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.native_config.action_token_fusion_mode == "none":
            return left_token, right_token
        if self._runtime_action_token_ablation_mode == "disable_fusion":
            return left_token, right_token
        legacy_no_remote = (
            self._runtime_action_token_ablation_mode is None
            and self._runtime_private_remote_kv_mode is None
            and self.native_config.remote_kv_mode == "no_remote"
        )
        allow_cross_arm = not legacy_no_remote
        if allow_cross_arm:
            self._record_pair_message("action_intent", left_token, right_token)
        action_ablation = self._runtime_action_token_ablation_mode or "none"
        if action_ablation == "random_remote":
            left_remote = self._random_like(right_token).to(device=left_token.device, dtype=left_token.dtype)
            right_remote = self._random_like(left_token).to(device=right_token.device, dtype=right_token.dtype)
        else:
            left_remote = right_token.to(device=left_token.device, dtype=left_token.dtype)
            right_remote = left_token.to(device=right_token.device, dtype=right_token.dtype)
        if self.native_config.action_token_fusion_mode == "paired_joint_transformer":
            if self.joint_action_token_transformer is None:
                raise RuntimeError("joint_action_token_transformer is required")
            if action_ablation == "random_remote":
                left_fused, _ = self.joint_action_token_transformer(
                    left_token,
                    left_remote,
                    allow_cross_arm=True,
                )
                _, right_fused = self.joint_action_token_transformer(
                    right_remote,
                    right_token,
                    allow_cross_arm=True,
                )
                return left_fused, right_fused
            return self.joint_action_token_transformer(
                left_token,
                right_token,
                allow_cross_arm=allow_cross_arm,
            )
        if self.left_action_token_fusion is None or self.right_action_token_fusion is None:
            raise RuntimeError(f"action token fusion modules are required for {self.native_config.action_token_fusion_mode}")

        if self.native_config.action_token_fusion_mode in ("paired_concat_project", "paired_residual_mlp") and not allow_cross_arm:
            left_remote = torch.zeros_like(left_token)
            right_remote = torch.zeros_like(right_token)

        left_in = torch.cat([left_token, left_remote], dim=-1)
        right_in = torch.cat([right_token, right_remote], dim=-1)

        left_fused = self.left_action_token_fusion(left_in.to(dtype=self._module_dtype(self.left_action_token_fusion)))
        right_fused = self.right_action_token_fusion(right_in.to(dtype=self._module_dtype(self.right_action_token_fusion)))
        if self.native_config.action_token_fusion_mode == "residual_exchange":
            return left_token + left_fused.to(dtype=left_token.dtype), right_token + right_fused.to(dtype=right_token.dtype)
        return left_fused.to(dtype=left_token.dtype), right_fused.to(dtype=right_token.dtype)

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        return getattr(module, "dtype", next(module.parameters()).dtype)

    def _action_loss(self, action_head: nn.Module, action_token: torch.Tensor, batch: BatchFeature) -> torch.Tensor:
        dtype = self._module_dtype(action_head)
        return action_head(
            batch["action"].to(dtype=dtype),
            action_token.to(dtype=dtype),
            batch["proprio"][:, 0, :].to(dtype=dtype),
        )

    def predict_normalized_actions(self, batch: BatchFeature | dict, *, cfg: float = 1.1, num_denoising_steps: int | None = None):
        left_batch = self._side_batch(batch, "left")
        right_batch = self._side_batch(batch, "right")
        outputs = self._three_way_forward(left_batch, right_batch)
        left_token = self._action_token_from_private(self.left.model, outputs["left_private"], outputs["left_private_modal_ids"])
        right_token = self._action_token_from_private(self.right.model, outputs["right_private"], outputs["right_private_modal_ids"])
        left_token, right_token = self._fuse_action_tokens(left_token, right_token)
        if self.native_config.decoder_mode == "shared_decoder_batched":
            assert self.shared_action_head is not None
            left_head = right_head = self.shared_action_head
        else:
            left_head = self.left.model.action_head
            right_head = self.right.model.action_head

        left_dtype = self._module_dtype(left_head)
        right_dtype = self._module_dtype(right_head)
        left_action = left_head.denoise(
            left_token.to(dtype=left_dtype),
            left_batch["proprio"][:, 0, :].to(dtype=left_dtype),
            denoising_steps=num_denoising_steps,
            cfg=cfg,
        ).reshape(-1, self.action_len, self.action_dim)
        right_action = right_head.denoise(
            right_token.to(dtype=right_dtype),
            right_batch["proprio"][:, 0, :].to(dtype=right_dtype),
            denoising_steps=num_denoising_steps,
            cfg=cfg,
        ).reshape(-1, self.action_len, self.action_dim)
        return torch.cat([left_action, right_action.to(device=left_action.device)], dim=-1)

    def forward(self, batch: BatchFeature | dict):
        left_batch = self._side_batch(batch, "left")
        right_batch = self._side_batch(batch, "right")
        outputs = self._three_way_forward(left_batch, right_batch)
        left_token = self._action_token_from_private(self.left.model, outputs["left_private"], outputs["left_private_modal_ids"])
        right_token = self._action_token_from_private(self.right.model, outputs["right_private"], outputs["right_private_modal_ids"])
        left_token, right_token = self._fuse_action_tokens(left_token, right_token)
        if self.native_config.decoder_mode == "shared_decoder_batched":
            assert self.shared_action_head is not None
            left_head = right_head = self.shared_action_head
        else:
            left_head = self.left.model.action_head
            right_head = self.right.model.action_head
        loss_l = self._action_loss(left_head, left_token, left_batch)
        loss_r = self._action_loss(right_head, right_token, right_batch)
        if self.native_config.loss_reduction == "mean":
            loss = (loss_l + loss_r.to(device=loss_l.device)) / 2.0
        else:
            loss = loss_l + loss_r.to(device=loss_l.device)
        return {
            "loss": loss,
            "loss_left": loss_l.detach(),
            "loss_right": loss_r.detach(),
        }

    def structural_summary(self, batch: BatchFeature | dict) -> dict[str, int | str | bool]:
        left = self._side_batch(batch, "left")
        right = self._side_batch(batch, "right")
        return {
            "common_strategy": self.native_config.common_strategy,
            "common_kv_mode": self.native_config.common_kv_mode,
            "common_fusion_mode": self.native_config.common_fusion_mode,
            "action_token_fusion_mode": self.native_config.action_token_fusion_mode,
            "decoder_mode": self.native_config.decoder_mode,
            "remote_kv_mode": self.native_config.remote_kv_mode,
            "remote_kv_scale": float(self.native_config.remote_kv_scale),
            "remote_kv_dropout_prob": float(self.native_config.remote_kv_dropout_prob),
            "left_common_tokens": int((left["ci_ids"] == 0).sum().item()),
            "left_private_tokens": int((left["ci_ids"] == 1).sum().item()),
            "right_common_tokens": int((right["ci_ids"] == 0).sum().item()),
            "right_private_tokens": int((right["ci_ids"] == 1).sum().item()),
            "left_action_head_trainable": any(p.requires_grad for p in self.left.model.action_head.parameters()),
            "right_action_head_trainable": any(p.requires_grad for p in self.right.model.action_head.parameters()),
            "action_transformer_layers": int(self.native_config.action_transformer_layers),
            "action_transformer_heads": int(self.native_config.action_transformer_heads),
        }

    def _extra_state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {}
        if self.common_layer_gate is not None:
            state["common_layer_gate"] = self.common_layer_gate.detach().cpu()
        if self.left_action_token_fusion is not None:
            state["left_action_token_fusion"] = {
                key: value.detach().cpu() for key, value in self.left_action_token_fusion.state_dict().items()
            }
        if self.right_action_token_fusion is not None:
            state["right_action_token_fusion"] = {
                key: value.detach().cpu() for key, value in self.right_action_token_fusion.state_dict().items()
            }
        if self.joint_action_token_transformer is not None:
            state["joint_action_token_transformer"] = {
                key: value.detach().cpu() for key, value in self.joint_action_token_transformer.state_dict().items()
            }
        return state

    def _load_extra_state_dict(self, state: dict[str, object]) -> None:
        if self.common_layer_gate is not None and "common_layer_gate" in state:
            tensor = state["common_layer_gate"]
            if torch.is_tensor(tensor):
                with torch.no_grad():
                    self.common_layer_gate.copy_(tensor.to(device=self.common_layer_gate.device, dtype=self.common_layer_gate.dtype))
        if self.left_action_token_fusion is not None and "left_action_token_fusion" in state:
            left_state = state["left_action_token_fusion"]
            if isinstance(self.left_action_token_fusion, ResidualMLPActionTokenFusion) and isinstance(left_state, dict) and "base.weight" not in left_state:
                self.left_action_token_fusion.load_base_linear_state_dict(left_state)
            else:
                self.left_action_token_fusion.load_state_dict(left_state)
        if self.right_action_token_fusion is not None and "right_action_token_fusion" in state:
            right_state = state["right_action_token_fusion"]
            if isinstance(self.right_action_token_fusion, ResidualMLPActionTokenFusion) and isinstance(right_state, dict) and "base.weight" not in right_state:
                self.right_action_token_fusion.load_base_linear_state_dict(right_state)
            else:
                self.right_action_token_fusion.load_state_dict(right_state)
        if self.joint_action_token_transformer is not None and "joint_action_token_transformer" in state:
            self.joint_action_token_transformer.load_state_dict(state["joint_action_token_transformer"])

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "commvla_native_v3_config.json").write_text(json.dumps(asdict(self.native_config), indent=2), encoding="utf-8")
        self.left.save_pretrained(path / "left_private_agent")
        self.right.save_pretrained(path / "right_private_agent")
        extra_state = self._extra_state_dict()
        if extra_state:
            torch.save(extra_state, path / EXTRA_STATE_FILE)

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: str | int = "cuda",
        right_device: str | int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        config_overrides: dict[str, object] | None = None,
    ) -> "CommVLANativeV3Pair":
        path = Path(directory)
        data = json.loads((path / "commvla_native_v3_config.json").read_text(encoding="utf-8"))
        data.pop("share_action_head", None)
        if config_overrides:
            data.update(config_overrides)
        data["left_agent_path"] = str(path / "left_private_agent")
        data["right_agent_path"] = str(path / "right_private_agent")
        model = cls(CommVLANativeV3Config(**data), device=device, right_device=right_device, dtype=dtype)
        extra_path = path / EXTRA_STATE_FILE
        if extra_path.exists():
            extra_state = torch.load(extra_path, map_location="cpu")
            model._load_extra_state_dict(extra_state)
        return model
