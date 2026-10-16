"""Core two-process split-device inference functions for CommVLA.

This script is intentionally narrow: it supports the v3.5 `right_shared`
common-KV mode and `full` private remote-KV exchange. Rank 0 loads the left
agent, rank 1 loads the right agent. The ranks exchange K/V tensors with
torch.distributed collectives at each layer and predict their own 10D action.
Rank 0 gathers both sides and writes a compact summary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from commvla.communication import (
    ChannelConfig,
    PeerCommunicationChannel,
    get_peer_channel,
    set_peer_channel,
)
from commvla.models.joint_action_transformer import JointActionTokenTransformer


def log(message: str) -> None:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    print(f"[rank {rank}] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@contextmanager
def inference_autocast(dtype: torch.dtype):
    if not torch.cuda.is_available() or dtype == torch.float32:
        with torch.amp.autocast("cuda", enabled=False):
            yield
    else:
        with torch.autocast("cuda", dtype=dtype):
            yield


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def normalize_state(stats: dict, unnorm_key: str, state: np.ndarray) -> np.ndarray:
    item = stats[unnorm_key]["proprio"]
    low = _array(item["q01"])
    high = _array(item["q99"])
    mask = np.asarray(item["mask"], dtype=bool)
    return np.where(mask, (state - low) * 2 / (high - low + 1e-6) - 1, state)


def unnormalize_action(stats: dict, unnorm_key: str, action: np.ndarray) -> np.ndarray:
    item = stats[unnorm_key]["action"]
    low = _array(item["q01"])
    high = _array(item["q99"])
    mask = np.asarray(item["mask"], dtype=bool)
    return np.where(mask, (action + 1) * (high - low + 1e-6) / 2 + low, action)


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


def make_side_batch(agent_model, stats: dict, unnorm_key: str, obs: dict, side: str, device, dtype: torch.dtype) -> BatchFeature:
    normalized_proprio = normalize_state(stats, unnorm_key, obs["ee_6d_pos"])
    if side == "left":
        wrist = obs["images"]["wrist_left"]
        proprio = normalized_proprio[:10]
    else:
        wrist = obs["images"]["wrist_right"]
        proprio = normalized_proprio[10:]
    inputs = agent_model.preprocess_inputs(
        obs["images"]["back"][np.newaxis, :].copy(),
        wrist[np.newaxis, :].copy(),
        obs["language_instruction"],
        action=None,
    )
    batch = BatchFeature()
    for key, value in inputs.items():
        batch[key] = value.unsqueeze(0) if torch.is_tensor(value) else value
    batch["proprio"] = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return _to_device(batch, device, dtype)


def select(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bsz = hidden.shape[0]
    return hidden[mask].reshape(bsz, -1, hidden.shape[-1])


def select_2d(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bsz = values.shape[0]
    return values[mask].reshape(bsz, -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def causal_mask(valid: torch.Tensor) -> torch.Tensor:
    bsz, seq_len = valid.shape
    device = valid.device
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    allowed = causal.unsqueeze(0) & valid.bool().unsqueeze(1).expand(-1, seq_len, -1)
    mask = torch.full((bsz, 1, seq_len, seq_len), torch.finfo(torch.float32).min, device=device)
    return mask.masked_fill(allowed.unsqueeze(1), 0.0)


def private_attention_mask(common_valid: torch.Tensor, local_valid: torch.Tensor, remote_valid: torch.Tensor) -> torch.Tensor:
    bsz, common_len = common_valid.shape
    local_len = local_valid.shape[1]
    remote_len = remote_valid.shape[1]
    device = local_valid.device
    common_allowed = common_valid.bool().unsqueeze(1).expand(-1, local_len, -1)
    order = torch.arange(local_len, device=device)
    tri = order[None, :] <= order[:, None]
    local_allowed = tri.unsqueeze(0) & local_valid.bool().unsqueeze(1).expand(-1, local_len, -1)
    remote_allowed = tri.unsqueeze(0) & remote_valid.bool().unsqueeze(1).expand(-1, local_len, -1)
    allowed = torch.cat([common_allowed, local_allowed, remote_allowed], dim=-1)
    total_len = common_len + local_len + remote_len
    mask = torch.full((bsz, 1, local_len, total_len), torch.finfo(torch.float32).min, device=device)
    return mask.masked_fill(allowed.unsqueeze(1), 0.0)


def project_qkv(layer, normed: torch.Tensor):
    attn = layer.self_attn
    shape = (*normed.shape[:-1], -1, attn.head_dim)
    q = attn.q_proj(normed).view(shape).transpose(1, 2)
    k = attn.k_proj(normed).view(shape).transpose(1, 2)
    v = attn.v_proj(normed).view(shape).transpose(1, 2)
    return q, k, v


def qkv_with_positions(layer, hidden: torch.Tensor, rotary_emb, position_ids: torch.Tensor):
    normed = layer.input_layernorm(hidden)
    q, k, v = project_qkv(layer, normed)
    q, k = apply_rotary_pos_emb(q, k, *rotary_emb(hidden, position_ids))
    return q, k, v


def attention_values(layer, query, key, value, mask: torch.Tensor) -> torch.Tensor:
    attn = layer.self_attn
    key_states = repeat_kv(key, attn.num_key_value_groups)
    value_states = repeat_kv(value, attn.num_key_value_groups)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * attn.scaling
    weights = weights + mask.to(dtype=weights.dtype)
    weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return output.reshape(output.shape[0], output.shape[1], -1)


def attention(layer, query, key, value, mask: torch.Tensor) -> torch.Tensor:
    return layer.self_attn.o_proj(attention_values(layer, query, key, value, mask))


def exchange_peer(
    tensor: torch.Tensor,
    *,
    group: str = "system",
    message_type: str = "untyped",
    layer_id: int = -1,
) -> torch.Tensor:
    channel = get_peer_channel()
    if channel is not None:
        return channel.exchange(
            tensor,
            group=group,
            message_type=message_type,
            layer_id=layer_id,
        )
    if dist.get_backend() == "gloo":
        send = tensor.detach().contiguous().cpu()
        gathered = [torch.empty_like(send) for _ in range(2)]
        dist.all_gather(gathered, send)
        return gathered[1 - dist.get_rank()].to(device=tensor.device, dtype=tensor.dtype)
    gathered = [torch.empty_like(tensor) for _ in range(2)]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered[1 - dist.get_rank()]


def private_remote_context(
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    local_valid: torch.Tensor,
    *,
    mode: str,
    layer_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build private peer context without communicating in no-remote mode."""
    if mode == "no_remote":
        return torch.zeros_like(local_k), torch.zeros_like(local_v), torch.zeros_like(local_valid)
    remote_k = exchange_peer(
        local_k,
        group="remote_kv",
        message_type="private_remote_k",
        layer_id=layer_id,
    )
    remote_v = exchange_peer(
        local_v,
        group="remote_kv",
        message_type="private_remote_v",
        layer_id=layer_id,
    )
    remote_valid = exchange_peer(
        local_valid,
        group="remote_kv",
        message_type="private_valid",
        layer_id=layer_id,
    )
    if mode == "random_remote":
        remote_k = random_like_without_advancing_rng(remote_k)
        remote_v = random_like_without_advancing_rng(remote_v)
    elif mode != "full":
        raise ValueError(f"Unsupported private_remote_kv_mode: {mode}")
    return remote_k, remote_v, remote_valid


def broadcast_from_right(tensor: torch.Tensor, device) -> torch.Tensor:
    if dist.get_backend() == "gloo":
        if dist.get_rank() == 1:
            out_cpu = tensor.detach().contiguous().cpu()
        else:
            out_cpu = torch.empty_like(tensor, device="cpu")
        dist.broadcast(out_cpu, src=1)
        return out_cpu.to(device=device, dtype=tensor.dtype)
    out = tensor.contiguous() if dist.get_rank() == 1 else torch.empty_like(tensor, device=device)
    dist.broadcast(out, src=1)
    return out


def action_token_from_private(agent_model, hidden_private: torch.Tensor, private_modal_ids: torch.Tensor) -> torch.Tensor:
    bsz = private_modal_ids.shape[0]
    action_token = hidden_private[private_modal_ids == 5].reshape(bsz, -1, agent_model.hidden_dim())
    return agent_model.agg(action_token)


def module_dtype(module: nn.Module) -> torch.dtype:
    return getattr(module, "dtype", next(module.parameters()).dtype)


def random_like_without_advancing_rng(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_cuda:
        state = torch.cuda.get_rng_state(tensor.device)
        value = torch.randn_like(tensor)
        torch.cuda.set_rng_state(state, tensor.device)
        return value
    state = torch.random.get_rng_state()
    value = torch.randn_like(tensor)
    torch.random.set_rng_state(state)
    return value


def _avg_with_peer(
    tensor: torch.Tensor,
    *,
    message_type: str,
    layer_id: int,
) -> torch.Tensor:
    peer = exchange_peer(
        tensor,
        group="common",
        message_type=message_type,
        layer_id=layer_id,
    )
    return (tensor + peer.to(device=tensor.device, dtype=tensor.dtype)) / 2.0


def _left_weighted_with_peer(
    tensor: torch.Tensor,
    common_layer_gate: torch.Tensor | None,
    layer_idx: int,
    message_type: str,
    enabled: bool = True,
) -> torch.Tensor:
    if not enabled:
        return tensor
    if common_layer_gate is None:
        return _avg_with_peer(tensor, message_type=message_type, layer_id=layer_idx)
    peer = exchange_peer(
        tensor,
        group="common",
        message_type=message_type,
        layer_id=layer_idx,
    ).to(device=tensor.device, dtype=tensor.dtype)
    gate = torch.sigmoid(common_layer_gate[layer_idx]).to(device=tensor.device, dtype=tensor.dtype)
    if dist.get_rank() == 0:
        return gate * tensor + (1.0 - gate) * peer
    return gate * peer + (1.0 - gate) * tensor


def common_sync_layer_ids(spec: str, num_layers: int) -> set[int]:
    if spec == "all":
        return set(range(num_layers))
    if spec == "late":
        return set(range(num_layers // 2, num_layers))
    if spec == "even":
        return {idx for idx in range(num_layers) if idx % 2 == 0}
    if spec == "late_even":
        return {idx for idx in range(num_layers // 2, num_layers) if idx % 2 == 0}
    if spec == "last6":
        return set(range(max(0, num_layers - 6), num_layers))
    raw = spec
    if spec.startswith("ids:"):
        raw = spec[4:]
    if spec.startswith("bits:"):
        bits = spec[5:].strip()
        if len(bits) != num_layers or any(bit not in "01" for bit in bits):
            raise ValueError(f"Expected a {num_layers}-bit common sync mask, got: {bits}")
        return {idx for idx, bit in enumerate(bits) if bit == "1"}
    if raw in {"", "none"}:
        return set()
    try:
        layer_ids = {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(f"Invalid common sync layer specification: {spec}") from exc
    if layer_ids and min(layer_ids) < 0 or layer_ids and max(layer_ids) >= num_layers:
        raise ValueError(f"Common sync layers must be within [0, {num_layers - 1}], got: {sorted(layer_ids)}")
    if layer_ids or raw == "none":
        return layer_ids
    raise ValueError(f"Unsupported common sync layer specification: {spec}")


def load_distributed_extra_modules(config: dict, checkpoint: Path, agent_model, device, dtype: torch.dtype, side: str) -> dict[str, object]:
    extra: dict[str, object] = {
        "common_fusion_mode": str(config.get("common_fusion_mode", "fixed_avg")),
        "action_token_fusion_mode": str(config.get("action_token_fusion_mode", "none")),
        "side": side,
        "common_layer_gate": None,
        "action_token_fusion": None,
        "joint_action_token_transformer": None,
    }
    state_path = checkpoint / "commvla_native_v3_extra.pt"
    state = torch.load(state_path, map_location="cpu") if state_path.exists() else {}

    if extra["common_fusion_mode"] == "layer_gate":
        num_layers = len(agent_model.text_backbone().model.layers)
        gate = state.get("common_layer_gate", torch.zeros(num_layers))
        if not torch.is_tensor(gate):
            raise TypeError("common_layer_gate in extra state must be a tensor")
        extra["common_layer_gate"] = gate.to(device=device, dtype=dtype)

    if extra["action_token_fusion_mode"] in ("residual_exchange", "paired_concat_project"):
        hidden_size = int(agent_model.hidden_dim())
        module = nn.Linear(hidden_size * 2, hidden_size).to(device=device, dtype=dtype)
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
        if extra["action_token_fusion_mode"] == "paired_concat_project":
            with torch.no_grad():
                module.weight[:, :hidden_size].copy_(torch.eye(hidden_size, device=device, dtype=dtype))
        key = "left_action_token_fusion" if side == "left" else "right_action_token_fusion"
        if key in state:
            module_state = state[key]
            module.load_state_dict(module_state)
            module.to(device=device, dtype=dtype)
        module.eval()
        extra["action_token_fusion"] = module
    elif extra["action_token_fusion_mode"] == "paired_joint_transformer":
        hidden_size = int(agent_model.hidden_dim())
        module = JointActionTokenTransformer(
            hidden_size,
            num_layers=int(config.get("action_transformer_layers", 3)),
            num_heads=int(config.get("action_transformer_heads", 8)),
            mlp_ratio=int(config.get("action_transformer_mlp_ratio", 2)),
            device=device,
            dtype=dtype,
        )
        module_state = state.get("joint_action_token_transformer")
        if not isinstance(module_state, dict):
            raise RuntimeError("paired_joint_transformer checkpoint is missing its shared state")
        module.load_state_dict(module_state)
        module.to(device=device, dtype=dtype)
        module.eval()
        extra["joint_action_token_transformer"] = module
    return extra


def fuse_distributed_action_token(token: torch.Tensor, extra_modules: dict[str, object]) -> torch.Tensor:
    mode = extra_modules.get("action_token_fusion_mode")
    if mode not in ("residual_exchange", "paired_concat_project", "paired_joint_transformer"):
        return token
    peer = exchange_peer(
        token,
        group="action",
        message_type="action_token",
    ).to(device=token.device, dtype=token.dtype)
    ablation_mode = str(extra_modules.get("action_token_ablation_mode", "none"))
    if ablation_mode == "disable_fusion":
        extra_modules["last_peer_action_token"] = peer.detach()
        return token
    if ablation_mode == "zero_remote":
        peer_for_fusion = torch.zeros_like(peer)
    elif ablation_mode == "random_remote":
        peer_for_fusion = random_like_without_advancing_rng(peer)
    elif ablation_mode == "stale_remote":
        cached = extra_modules.get("last_peer_action_token")
        peer_for_fusion = cached.to(device=peer.device, dtype=peer.dtype) if torch.is_tensor(cached) else torch.zeros_like(peer)
    elif ablation_mode == "none":
        peer_for_fusion = peer
    else:
        raise ValueError(f"Unsupported action_token_ablation_mode: {ablation_mode}")
    extra_modules["last_peer_action_token"] = peer.detach()
    if mode == "paired_joint_transformer":
        module = extra_modules.get("joint_action_token_transformer")
        if not isinstance(module, JointActionTokenTransformer):
            raise RuntimeError("joint_action_token_transformer is required for paired_joint_transformer")
        side = str(extra_modules.get("side"))
        if side == "left":
            fused, _ = module(token, peer_for_fusion, allow_cross_arm=True)
        elif side == "right":
            _, fused = module(peer_for_fusion, token, allow_cross_arm=True)
        else:
            raise ValueError(f"Unsupported distributed agent side: {side}")
        return fused.to(dtype=token.dtype)
    module = extra_modules.get("action_token_fusion")
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"action_token_fusion module is required for {mode}")
    fused_input = torch.cat([token, peer_for_fusion], dim=-1)
    fused = module(fused_input.to(dtype=module_dtype(module)))
    if mode == "residual_exchange":
        return token + fused.to(dtype=token.dtype)
    return fused.to(dtype=token.dtype)


def distributed_v35_forward(
    agent_model,
    batch: BatchFeature,
    side: str,
    common_kv_mode: str = "right_shared",
    common_layer_gate: torch.Tensor | None = None,
    private_remote_kv_mode: str = "full",
    common_sync_strategy: str = "full",
    common_sync_layers: str = "all",
    hidden_trace: list[dict[str, object]] | None = None,
    prepared_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    rank = dist.get_rank()
    if prepared_embeddings is None:
        hidden_full, mask_full = agent_model.prepare_embeds(batch)
    else:
        hidden_full, mask_full = prepared_embeddings
    common_mask = batch["ci_ids"] == 0
    private_mask = batch["ci_ids"] == 1
    hidden_c = select(hidden_full, common_mask)
    hidden_p = select(hidden_full, private_mask)
    common_valid = select_2d(mask_full, common_mask).to(device=hidden_c.device)
    private_valid = select_2d(mask_full, private_mask).to(device=hidden_p.device)
    private_modal = select_2d(batch["modal_ids"], private_mask).to(device=hidden_p.device)

    bsz = hidden_p.shape[0]
    common_len = hidden_c.shape[1]
    private_len = hidden_p.shape[1]
    pos_c = torch.arange(common_len, device=hidden_c.device).unsqueeze(0).expand(bsz, -1)
    pos_p = (common_len + torch.arange(private_len, device=hidden_p.device)).unsqueeze(0).expand(bsz, -1)
    layers = agent_model.text_backbone().model.layers
    sync_layer_ids = common_sync_layer_ids(common_sync_layers, len(layers))
    rotary = agent_model.text_backbone().model.rotary_emb
    common_attn_mask = causal_mask(common_valid)

    if common_kv_mode == "sync_qkv_avg_mlp_avg":
        peer_hidden_c = exchange_peer(
            hidden_c,
            group="common",
            message_type="common_initial_hidden",
        )
        hidden_c = (hidden_c + peer_hidden_c.to(device=hidden_c.device, dtype=hidden_c.dtype)) / 2.0
        peer_common_valid = exchange_peer(
            common_valid,
            group="common",
            message_type="common_valid",
        )
        common_valid_sync = (common_valid.bool() & peer_common_valid.to(device=common_valid.device).bool()).to(
            dtype=common_valid.dtype
        )
        common_attn_mask_sync = causal_mask(common_valid_sync)

        for layer_idx, layer in enumerate(layers):
            if layer_idx == 0 or (layer_idx + 1) % 8 == 0 or layer_idx == len(layers) - 1:
                log(f"forward layer {layer_idx + 1}/{len(layers)}")

            sync_this_layer = layer_idx in sync_layer_ids
            residual_c = hidden_c
            norm_c = _left_weighted_with_peer(
                layer.input_layernorm(hidden_c), common_layer_gate, layer_idx, "common_norm",
                enabled=sync_this_layer and common_sync_strategy == "full",
            )
            q_c, k_c, v_c = project_qkv(layer, norm_c)
            q_c = _left_weighted_with_peer(
                q_c, common_layer_gate, layer_idx, "common_q",
                enabled=sync_this_layer and common_sync_strategy == "full",
            )
            k_c = _left_weighted_with_peer(
                k_c, common_layer_gate, layer_idx, "common_k",
                enabled=sync_this_layer and common_sync_strategy == "full",
            )
            v_c = _left_weighted_with_peer(
                v_c, common_layer_gate, layer_idx, "common_v",
                enabled=sync_this_layer and common_sync_strategy == "full",
            )
            q_c, k_c = apply_rotary_pos_emb(q_c, k_c, *rotary(hidden_c, pos_c))
            common_attn_values = attention_values(layer, q_c, k_c, v_c, common_attn_mask_sync)
            common_attn = _left_weighted_with_peer(
                layer.self_attn.o_proj(common_attn_values),
                common_layer_gate,
                layer_idx,
                "common_attention_output",
                enabled=sync_this_layer and common_sync_strategy == "full",
            )
            hidden_c = residual_c + common_attn

            residual_c = hidden_c
            mlp_norm_c = _left_weighted_with_peer(
                layer.post_attention_layernorm(hidden_c),
                common_layer_gate,
                layer_idx,
                "common_mlp_norm",
                enabled=sync_this_layer and common_sync_strategy == "full",
            )
            mlp_c = _left_weighted_with_peer(
                layer.mlp(mlp_norm_c),
                common_layer_gate,
                layer_idx,
                "common_mlp_output",
                enabled=sync_this_layer and common_sync_strategy == "full",
            )
            hidden_c = residual_c + mlp_c
            if hidden_trace is not None:
                trace_item: dict[str, object] = {
                    "layer_id": layer_idx,
                    "sync_enabled": bool(sync_this_layer),
                    "pre_sync": hidden_c.detach().clone(),
                }
            if common_sync_strategy == "hidden_only":
                if sync_this_layer:
                    hidden_c = _avg_with_peer(
                        hidden_c,
                        message_type="common_hidden",
                        layer_id=layer_idx,
                    )
            elif common_sync_strategy != "full":
                raise ValueError(f"Unsupported common_sync_strategy: {common_sync_strategy}")
            if hidden_trace is not None:
                trace_item["post_sync"] = hidden_c.detach().clone()
                hidden_trace.append(trace_item)

            residual_p = hidden_p
            q_p, k_p, v_p = qkv_with_positions(layer, hidden_p, rotary, pos_p)
            remote_k, remote_v, remote_valid = private_remote_context(
                k_p,
                v_p,
                private_valid,
                mode=private_remote_kv_mode,
                layer_id=layer_idx,
            )

            k_for_p = torch.cat(
                [k_c.to(device=k_p.device, dtype=k_p.dtype), k_p, remote_k.to(device=k_p.device, dtype=k_p.dtype)],
                dim=2,
            )
            v_for_p = torch.cat(
                [v_c.to(device=v_p.device, dtype=v_p.dtype), v_p, remote_v.to(device=v_p.device, dtype=v_p.dtype)],
                dim=2,
            )
            mask_for_p = private_attention_mask(common_valid_sync.to(device=hidden_p.device), private_valid, remote_valid.to(device=hidden_p.device))
            hidden_p = residual_p + attention(layer, q_p, k_for_p, v_for_p, mask_for_p)
            hidden_p = hidden_p + layer.mlp(layer.post_attention_layernorm(hidden_p))

        hidden_p = agent_model.text_backbone().model.norm(hidden_p)
        return hidden_p, private_modal

    if common_kv_mode not in {"right_shared", "local", "left_shared", "avg_shared"}:
        raise ValueError(f"Unsupported distributed common_kv_mode: {common_kv_mode}")

    for layer_idx, layer in enumerate(layers):
        if layer_idx == 0 or (layer_idx + 1) % 8 == 0 or layer_idx == len(layers) - 1:
            log(f"forward layer {layer_idx + 1}/{len(layers)}")
        residual_c = hidden_c
        q_c, k_c, v_c = qkv_with_positions(layer, hidden_c, rotary, pos_c)
        hidden_c = residual_c + attention(layer, q_c, k_c, v_c, common_attn_mask)
        hidden_c = hidden_c + layer.mlp(layer.post_attention_layernorm(hidden_c))

        residual_p = hidden_p
        q_p, k_p, v_p = qkv_with_positions(layer, hidden_p, rotary, pos_p)

        remote_k, remote_v, remote_valid = private_remote_context(
            k_p,
            v_p,
            private_valid,
            mode=private_remote_kv_mode,
            layer_id=layer_idx,
        )

        if common_kv_mode == "right_shared":
            common_k = broadcast_from_right(k_c, hidden_p.device)
            common_v = broadcast_from_right(v_c, hidden_p.device)
        elif common_kv_mode == "left_shared":
            if dist.get_backend() == "gloo":
                if dist.get_rank() == 0:
                    common_k = k_c
                    common_v = v_c
                else:
                    common_k = torch.empty_like(k_c)
                    common_v = torch.empty_like(v_c)
                common_k_cpu = common_k.detach().contiguous().cpu()
                common_v_cpu = common_v.detach().contiguous().cpu()
                dist.broadcast(common_k_cpu, src=0)
                dist.broadcast(common_v_cpu, src=0)
                common_k = common_k_cpu.to(device=hidden_p.device, dtype=k_c.dtype)
                common_v = common_v_cpu.to(device=hidden_p.device, dtype=v_c.dtype)
            else:
                common_k = k_c.contiguous() if dist.get_rank() == 0 else torch.empty_like(k_c, device=hidden_p.device)
                common_v = v_c.contiguous() if dist.get_rank() == 0 else torch.empty_like(v_c, device=hidden_p.device)
                dist.broadcast(common_k, src=0)
                dist.broadcast(common_v, src=0)
        elif common_kv_mode == "avg_shared":
            common_k = _avg_with_peer(k_c, message_type="common_k", layer_id=layer_idx)
            common_v = _avg_with_peer(v_c, message_type="common_v", layer_id=layer_idx)
        else:
            common_k = k_c
            common_v = v_c

        k_for_p = torch.cat([common_k.to(device=k_p.device, dtype=k_p.dtype), k_p, remote_k.to(device=k_p.device)], dim=2)
        v_for_p = torch.cat([common_v.to(device=v_p.device, dtype=v_p.dtype), v_p, remote_v.to(device=v_p.device)], dim=2)
        mask_for_p = private_attention_mask(common_valid, private_valid, remote_valid.to(device=hidden_p.device))
        hidden_p = residual_p + attention(layer, q_p, k_for_p, v_for_p, mask_for_p)
        hidden_p = hidden_p + layer.mlp(layer.post_attention_layernorm(hidden_p))

    hidden_p = agent_model.text_backbone().model.norm(hidden_p)
    return hidden_p, private_modal


def main() -> None:
    from twinvla.model.singlevla import SingleVLA

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--unnorm-key", default="aloha_handover_box")
    parser.add_argument("--task-name", default="aloha_handover_box")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=1.1)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--benchmark-id", type=int, default=0)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument(
        "--action-token-ablation-mode",
        choices=["none", "zero_remote", "random_remote", "stale_remote", "disable_fusion"],
        default="none",
    )
    parser.add_argument("--private-remote-kv-mode", choices=["full", "no_remote", "random_remote"], default="full")
    parser.add_argument("--comm-profile-dir", default=None)
    parser.add_argument("--comm-profile-mode", choices=["none", "bytes", "timing"], default="none")
    parser.add_argument("--comm-action-refresh", choices=["1", "2", "4", "8", "initial"], default="1")
    parser.add_argument("--comm-common-refresh", choices=["1", "2", "4", "8", "initial"], default="1")
    parser.add_argument("--comm-remote-kv-refresh", choices=["1", "2", "4", "8", "initial"], default="1")
    codec_choices = [
        "raw", "int8", "int8_packed", "fp8_e4m3", "fp8_e5m2", "int4",
        "mixed_i4_a", "mixed_i4_b", "delta_int8", "delta_int4",
    ]
    parser.add_argument("--comm-action-codec", choices=codec_choices, default="raw")
    parser.add_argument("--comm-common-codec", choices=codec_choices, default="raw")
    parser.add_argument("--comm-remote-kv-codec", choices=codec_choices, default="raw")
    parser.add_argument("--comm-delta-keyframe", type=int, default=4)
    parser.add_argument("--comm-common-sync-strategy", choices=["full", "hidden_only"], default="full")
    parser.add_argument(
        "--comm-common-sync-layers",
        default="all",
        help="Named mask (all/late/even/late_even/last6), ids:0,2,..., or bits:<24 bits>.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = parse_dtype(args.dtype)
    set_seed(args.seed)
    log(f"initialized process group on local_rank={local_rank}, device={device}, dtype={args.dtype}")

    communication_enabled = bool(
        args.comm_profile_dir
        or args.comm_profile_mode != "none"
        or args.comm_action_refresh != "1"
        or args.comm_common_refresh != "1"
        or args.comm_remote_kv_refresh != "1"
        or args.comm_action_codec != "raw"
        or args.comm_common_codec != "raw"
        or args.comm_remote_kv_codec != "raw"
        or args.comm_common_sync_strategy != "full"
        or args.comm_common_sync_layers != "all"
    )
    channel = None
    if communication_enabled:
        channel = PeerCommunicationChannel(
            ChannelConfig.from_values(
                action_refresh=args.comm_action_refresh,
                common_refresh=args.comm_common_refresh,
                remote_kv_refresh=args.comm_remote_kv_refresh,
                action_codec=args.comm_action_codec,
                common_codec=args.comm_common_codec,
                remote_kv_codec=args.comm_remote_kv_codec,
                delta_keyframe=args.comm_delta_keyframe,
                profile_mode=args.comm_profile_mode,
            ),
            Path(args.comm_profile_dir) if args.comm_profile_dir else None,
        )
        channel.begin_round(args.benchmark_id, 0)
        set_peer_channel(channel)

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "commvla_native_v3_config.json").read_text(encoding="utf-8"))
    args.private_remote_kv_mode = str(config.get("remote_kv_mode", "full"))
    common_kv_mode = str(config.get("common_kv_mode", "right_shared"))
    agent_path = checkpoint / ("left_private_agent" if rank == 0 else "right_private_agent")
    side = "left" if rank == 0 else "right"
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text(encoding="utf-8"))

    # Match the validated TwinVLA-agentized path: each rank constructs the same
    # deterministic benchmark observation locally. Broadcasting image payloads as
    # Python objects over NCCL can block on some configurations.
    log("creating local Tabletop benchmark observation")
    import tabletop

    # Tabletop-Sim writes a process-global aloha_temp.xml while constructing an
    # environment. Serialize construction so the two ranks cannot race on it.
    env = None
    for owner_rank in range(dist.get_world_size()):
        if rank == owner_rank:
            env = tabletop.env(args.task_name, "ee_6d_pos")
        dist.barrier()
    assert env is not None
    ts = env.reset()
    ts = env.task.benchmark_init(env.physics, args.benchmark_id)
    raw_obs = ts.observation
    obs = {
        "ee_6d_pos": np.asarray(raw_obs["ee_6d_pos"], dtype=np.float32),
        "language_instruction": raw_obs["language_instruction"],
        "images": {
            "back": np.asarray(raw_obs["images"]["back"]),
            "wrist_left": np.asarray(raw_obs["images"]["wrist_left"]),
            "wrist_right": np.asarray(raw_obs["images"]["wrist_right"]),
        },
    }
    log("local observation ready")

    load_started = time.perf_counter()
    log(f"loading {side} agent from {agent_path}")
    agent = SingleVLA(pretrained_path=str(agent_path), device=device, dtype=dtype).model
    agent.config.use_cache = False
    agent.eval()
    extra_modules = load_distributed_extra_modules(config, checkpoint, agent, device, dtype, side)
    extra_modules["action_token_ablation_mode"] = args.action_token_ablation_mode
    load_seconds = time.perf_counter() - load_started
    log(f"loaded {side} agent in {load_seconds:.2f}s")
    log("building side batch")
    batch = make_side_batch(agent, stats, args.unnorm_key, obs, side, device, dtype)
    log("side batch ready")

    dist.barrier()
    started = time.perf_counter()
    log("starting distributed forward/action decode")
    with torch.no_grad(), inference_autocast(dtype):
        set_seed(args.seed + 30000 + rank)
        hidden_private, private_modal = distributed_v35_forward(
            agent,
            batch,
            side,
            common_kv_mode=common_kv_mode,
            common_layer_gate=extra_modules.get("common_layer_gate"),
            private_remote_kv_mode=args.private_remote_kv_mode,
            common_sync_strategy=args.comm_common_sync_strategy,
            common_sync_layers=args.comm_common_sync_layers,
        )
        token = action_token_from_private(agent, hidden_private, private_modal)
        token = fuse_distributed_action_token(token, extra_modules)
        head = agent.action_head
        head_dtype = module_dtype(head)
        normalized_side_action = head.denoise(
            token.to(dtype=head_dtype),
            batch["proprio"][:, 0, :].to(dtype=head_dtype),
            denoising_steps=args.num_denoising_steps,
            cfg=args.cfg,
        ).reshape(-1, int(agent.config.action_len), int(agent.config.action_dim))
    torch.cuda.synchronize(device)
    dist.barrier()
    infer_seconds = time.perf_counter() - started
    log(f"finished distributed forward/action decode in {infer_seconds:.2f}s")

    if dist.get_backend() == "gloo":
        local_for_gather = normalized_side_action.detach().contiguous().cpu()
        gathered = [torch.empty_like(local_for_gather) for _ in range(2)]
        dist.all_gather(gathered, local_for_gather)
    else:
        gathered = [torch.empty_like(normalized_side_action) for _ in range(2)]
        dist.all_gather(gathered, normalized_side_action.contiguous())
    local_finite = bool(torch.isfinite(normalized_side_action).all().item())
    finite_flags = [None, None]
    dist.all_gather_object(finite_flags, local_finite)
    latency_items = [None, None]
    dist.all_gather_object(latency_items, {"rank": rank, "side": side, "load_seconds": load_seconds, "infer_seconds": infer_seconds})

    if rank == 0:
        normalized = torch.cat([gathered[0], gathered[1].to(device=gathered[0].device)], dim=-1).detach().cpu().float().numpy()
        action = unnormalize_action(stats, args.unnorm_key, normalized)
        summary = {
            "checkpoint": str(checkpoint),
            "dtype": args.dtype,
            "common_kv_mode": common_kv_mode,
            "common_fusion_mode": extra_modules.get("common_fusion_mode"),
            "action_token_fusion_mode": extra_modules.get("action_token_fusion_mode"),
            "action_token_ablation_mode": args.action_token_ablation_mode,
            "private_remote_kv_mode": args.private_remote_kv_mode,
            "communication": {
                "enabled": communication_enabled,
                "profile_mode": args.comm_profile_mode,
                "action_refresh": args.comm_action_refresh,
                "common_refresh": args.comm_common_refresh,
                "remote_kv_refresh": args.comm_remote_kv_refresh,
                "action_codec": args.comm_action_codec,
                "common_codec": args.comm_common_codec,
                "remote_kv_codec": args.comm_remote_kv_codec,
                "delta_keyframe": args.comm_delta_keyframe,
                "common_sync_strategy": args.comm_common_sync_strategy,
                "common_sync_layers": args.comm_common_sync_layers,
            },
            "benchmark_id": args.benchmark_id,
            "finite_flags": finite_flags,
            "normalized_finite": bool(np.isfinite(normalized).all()),
            "normalized_nan_count": int(np.isnan(normalized).sum()),
            "action_finite": bool(np.isfinite(action).all()),
            "action_nan_count": int(np.isnan(action).sum()),
            "normalized_min": float(np.nanmin(normalized)),
            "normalized_max": float(np.nanmax(normalized)),
            "normalized_action": normalized.tolist(),
            "action": action.tolist(),
            "latency": latency_items,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    if channel is not None:
        channel.close()
        set_peer_channel(None)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
