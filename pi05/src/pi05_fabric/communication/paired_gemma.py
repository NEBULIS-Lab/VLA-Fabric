"""Functional paired execution of two independent OpenPI Gemma stacks."""

from __future__ import annotations

import math
import re
from typing import Any, NamedTuple, Sequence

import einops
import jax
import jax.numpy as jnp

from pi05_fabric.communication.raw_common import RAW_FUSION_POINTS
from pi05_fabric.data.aloha_dual_agent import COMMON, PRIVATE


class PairedGemmaTrace(NamedTuple):
    raw_event_counts: jax.Array
    remote_private_tokens: int
    remote_action_tokens: int = 0
    remote_action_residual_tokens: int = 0
    transmitted_queries: int = 0
    transmitted_raw_observations: int = 0


def action_suffix_kv(key, value, *, action_horizon: int):
    """Select only action-owned suffix K/V for peer transmission."""
    if key.shape != value.shape:
        raise ValueError("action key and value tensors must have equal shapes")
    if action_horizon <= 0 or action_horizon > key.shape[1]:
        raise ValueError("action_horizon must select a non-empty suffix")
    return key[:, -action_horizon:], value[:, -action_horizon:]


def _fuse_common(left, right, ownership, enabled):
    if not enabled:
        return left, right
    owner = jnp.asarray(ownership)
    if owner.ndim == 1:
        mask = owner[None, :, None] == COMMON
    elif owner.ndim == 2:
        mask = owner[:, :, None] == COMMON
    else:
        raise ValueError("ownership must have shape [tokens] or [batch, tokens]")
    while mask.ndim < left.ndim:
        mask = mask[..., None]
    average = (left + right) / 2
    return jnp.where(mask, average, left), jnp.where(mask, average, right)


def _rms_norm(x, params, cond):
    dtype = x.dtype
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normalized = x * jax.lax.rsqrt(variance + 1e-6)
    if cond is None:
        return (normalized * (1 + params["scale"])).astype(dtype), None
    dense = params["Dense_0"]
    modulation = jnp.dot(cond.astype(dtype), dense["kernel"].astype(dtype)) + dense["bias"].astype(dtype)
    scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
    return (normalized * (1 + scale) + shift).astype(dtype), gate


def _lora_einsum(equation, x, params, config):
    result = jnp.einsum(equation, x, params["w"].astype(x.dtype))
    if "lora_a" not in params:
        return result
    match = re.fullmatch(r"(.*),(.*)->(.*)", equation)
    if match is None:
        raise ValueError(f"unsupported einsum equation: {equation}")
    lhs, rhs, output = match.groups()
    axis_a, axis_b = config.axes
    a_label, b_label = rhs[axis_a], rhs[axis_b]
    rank_label = config.label
    a_rhs = rhs.replace(b_label, rank_label)
    a_output = output.replace(b_label, rank_label)
    b_rhs = rhs.replace(a_label, rank_label)
    low_rank = jnp.einsum(f"{lhs},{a_rhs}->{a_output}", x, params["lora_a"].astype(x.dtype))
    low_rank = jnp.einsum(f"{a_output},{b_rhs}->{output}", low_rank, params["lora_b"].astype(x.dtype))
    return result + low_rank * config.scaling_value


def _project_qkv(x, params, config, suffix):
    tag = "_1" if suffix else ""
    lora_config = config.lora_configs.get("attn")
    q = _lora_einsum("BTD,NDH->BTNH", x, params[f"q_einsum{tag}"], lora_config)
    kv = _lora_einsum("BSD,2KDH->2BSKH", x, params[f"kv_einsum{tag}"], lora_config)
    return q, kv[0], kv[1]


def _apply_rope(x, positions, max_wavelength=10_000):
    exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sine, cosine = jnp.sin(radians), jnp.cos(radians)
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([first * cosine - second * sine, second * cosine + first * sine], axis=-1).astype(x.dtype)


def _attention(query, key, value, mask, config):
    query = einops.rearrange(
        query, "B T (K G) H -> B T K G H", K=config.num_kv_heads
    )
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", query, key, preferred_element_type=jnp.float32)
    logits = jnp.where(mask[:, None, None, :, :], logits, -2.3819763e38)
    probabilities = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probabilities, value)
    return einops.rearrange(encoded, "B T K G H -> B T (K G) H")


def _residual_action_update(query, key, value, mask, kernel, gate, config):
    encoded = _attention(query, key, value, mask, config)
    flattened = einops.rearrange(encoded, "B T N H -> B T (N H)")
    projected = jnp.einsum(
        "BTD,DF->BTF", flattened, kernel.astype(flattened.dtype)
    )
    return projected * gate.astype(projected.dtype)


def _output_projection(encoded, params, config, suffix):
    tag = "_1" if suffix else ""
    return _lora_einsum(
        "BTNH,NHD->BTD",
        encoded,
        params[f"attn_vec_einsum{tag}"],
        config.lora_configs.get("attn"),
    )


def _feed_forward(x, params, config):
    dtype = x.dtype
    gating = params["gating_einsum"].astype(dtype)
    linear = params["linear"].astype(dtype)
    if "gating_einsum_lora_a" in params:
        gating = gating
        gate_lora = params["gating_einsum_lora_a"].astype(dtype), params["gating_einsum_lora_b"].astype(dtype)
        linear_lora = params["linear_lora_a"].astype(dtype), params["linear_lora_b"].astype(dtype)
    else:
        gate_lora = linear_lora = None

    def dot(value, weight, low_rank):
        output = jnp.dot(value, weight)
        if low_rank is not None:
            output = output + jnp.dot(jnp.dot(value, low_rank[0]), low_rank[1])
        return output

    gate = jax.nn.gelu(dot(x, gating[0], None if gate_lora is None else (gate_lora[0][0], gate_lora[1][0])))
    value = dot(x, gating[1], None if gate_lora is None else (gate_lora[0][1], gate_lora[1][1]))
    return dot(gate * value, linear, linear_lora).astype(dtype)


def _slice_layer(tree, layer):
    return jax.tree.map(lambda value: value[layer], tree)


def _batched_ownership(ownership, *, batch_size: int, token_count: int):
    owner = jnp.asarray(ownership)
    if owner.ndim == 1:
        if owner.shape[0] != token_count:
            raise ValueError("ownership token count does not match the prefix")
        return jnp.broadcast_to(owner[None], (batch_size, token_count))
    if owner.ndim == 2 and owner.shape == (batch_size, token_count):
        return owner
    raise ValueError(
        "ownership must have shape [tokens] or [batch, tokens] matching the prefix"
    )


def _run_layer_pair(
    left_xs,
    right_xs,
    left_params,
    right_params,
    configs,
    left_positions,
    right_positions,
    left_mask,
    right_mask,
    left_cond,
    right_cond,
    ownership,
    common_enabled,
    private_kv_enabled,
    remote_action_kv_enabled,
    action_horizon,
    remote_action_residual_enabled,
    left_residual_params,
    right_residual_params,
):
    left_prefix, left_suffix = left_xs
    right_prefix, right_suffix = right_xs
    prefix_lengths = left_prefix.shape[1], right_prefix.shape[1]

    left_norm_prefix, left_gate_prefix = _rms_norm(left_prefix, left_params["pre_attention_norm"], None)
    right_norm_prefix, right_gate_prefix = _rms_norm(right_prefix, right_params["pre_attention_norm"], None)
    left_norm_prefix, right_norm_prefix = _fuse_common(
        left_norm_prefix, right_norm_prefix, ownership, common_enabled
    )
    left_norm_suffix, left_gate_suffix = _rms_norm(left_suffix, left_params["pre_attention_norm_1"], left_cond[1])
    right_norm_suffix, right_gate_suffix = _rms_norm(right_suffix, right_params["pre_attention_norm_1"], right_cond[1])

    left_qp, left_kp, left_vp = _project_qkv(left_norm_prefix, left_params["attn"], configs[0], False)
    right_qp, right_kp, right_vp = _project_qkv(right_norm_prefix, right_params["attn"], configs[0], False)
    left_qp, right_qp = _fuse_common(left_qp, right_qp, ownership, common_enabled)
    left_kp, right_kp = _fuse_common(left_kp, right_kp, ownership, common_enabled)
    left_vp, right_vp = _fuse_common(left_vp, right_vp, ownership, common_enabled)
    left_qs, left_ks, left_vs = _project_qkv(left_norm_suffix, left_params["attn"], configs[1], True)
    right_qs, right_ks, right_vs = _project_qkv(right_norm_suffix, right_params["attn"], configs[1], True)

    lp, rp = prefix_lengths
    left_qp = _apply_rope(left_qp, left_positions[:, :lp]) * configs[0].head_dim**-0.5
    left_qs = _apply_rope(left_qs, left_positions[:, lp:]) * configs[0].head_dim**-0.5
    right_qp = _apply_rope(right_qp, right_positions[:, :rp]) * configs[0].head_dim**-0.5
    right_qs = _apply_rope(right_qs, right_positions[:, rp:]) * configs[0].head_dim**-0.5
    left_kp, left_ks = _apply_rope(left_kp, left_positions[:, :lp]), _apply_rope(left_ks, left_positions[:, lp:])
    right_kp, right_ks = _apply_rope(right_kp, right_positions[:, :rp]), _apply_rope(right_ks, right_positions[:, rp:])

    left_local_k, left_local_v = jnp.concatenate([left_kp, left_ks], axis=1), jnp.concatenate([left_vp, left_vs], axis=1)
    right_local_k, right_local_v = jnp.concatenate([right_kp, right_ks], axis=1), jnp.concatenate([right_vp, right_vs], axis=1)
    left_encoded_prefix = _attention(left_qp, left_local_k, left_local_v, left_mask[:, :lp], configs[0])
    right_encoded_prefix = _attention(right_qp, right_local_k, right_local_v, right_mask[:, :rp], configs[0])

    left_suffix_mask, right_suffix_mask = left_mask[:, lp:], right_mask[:, rp:]
    left_suffix_k, left_suffix_v = left_local_k, left_local_v
    right_suffix_k, right_suffix_v = right_local_k, right_local_v
    if private_kv_enabled:
        if lp != rp:
            raise ValueError("left and right prefixes must use the same padded length")
        owner = _batched_ownership(ownership, batch_size=left_prefix.shape[0], token_count=lp)
        private_mask = owner == PRIVATE
        # Keep a fixed padded transport shape for JIT batching. Non-Private peer
        # slots are masked before softmax and cannot influence the local suffix.
        left_remote_k, left_remote_v = right_kp, right_vp
        right_remote_k, right_remote_v = left_kp, left_vp
        left_suffix_k, left_suffix_v = jnp.concatenate([left_local_k, left_remote_k], axis=1), jnp.concatenate([left_local_v, left_remote_v], axis=1)
        right_suffix_k, right_suffix_v = jnp.concatenate([right_local_k, right_remote_k], axis=1), jnp.concatenate([right_local_v, right_remote_v], axis=1)
        left_remote_mask = jnp.logical_and(right_mask[:, rp, :rp], private_mask)
        right_remote_mask = jnp.logical_and(left_mask[:, lp, :lp], private_mask)
        left_suffix_mask = jnp.concatenate([left_suffix_mask, jnp.broadcast_to(left_remote_mask[:, None], (left_remote_mask.shape[0], left_suffix.shape[1], left_remote_mask.shape[1]))], axis=-1)
        right_suffix_mask = jnp.concatenate([right_suffix_mask, jnp.broadcast_to(right_remote_mask[:, None], (right_remote_mask.shape[0], right_suffix.shape[1], right_remote_mask.shape[1]))], axis=-1)

    if remote_action_kv_enabled:
        left_remote_action_k, left_remote_action_v = action_suffix_kv(
            right_ks, right_vs, action_horizon=action_horizon
        )
        right_remote_action_k, right_remote_action_v = action_suffix_kv(
            left_ks, left_vs, action_horizon=action_horizon
        )
        left_suffix_k = jnp.concatenate([left_suffix_k, left_remote_action_k], axis=1)
        left_suffix_v = jnp.concatenate([left_suffix_v, left_remote_action_v], axis=1)
        right_suffix_k = jnp.concatenate([right_suffix_k, right_remote_action_k], axis=1)
        right_suffix_v = jnp.concatenate([right_suffix_v, right_remote_action_v], axis=1)

        left_peer_action_valid = jnp.any(
            right_mask[:, rp:, rp:][:, :, -action_horizon:], axis=1
        )
        right_peer_action_valid = jnp.any(
            left_mask[:, lp:, lp:][:, :, -action_horizon:], axis=1
        )
        left_suffix_mask = jnp.concatenate(
            [
                left_suffix_mask,
                jnp.broadcast_to(
                    left_peer_action_valid[:, None],
                    (left_peer_action_valid.shape[0], left_suffix.shape[1], action_horizon),
                ),
            ],
            axis=-1,
        )
        right_suffix_mask = jnp.concatenate(
            [
                right_suffix_mask,
                jnp.broadcast_to(
                    right_peer_action_valid[:, None],
                    (right_peer_action_valid.shape[0], right_suffix.shape[1], action_horizon),
                ),
            ],
            axis=-1,
        )

    left_action_residual = right_action_residual = None
    if remote_action_residual_enabled:
        left_remote_action_k, left_remote_action_v = action_suffix_kv(
            right_ks, right_vs, action_horizon=action_horizon
        )
        right_remote_action_k, right_remote_action_v = action_suffix_kv(
            left_ks, left_vs, action_horizon=action_horizon
        )
        left_peer_action_valid = jnp.any(
            right_mask[:, rp:, rp:][:, :, -action_horizon:], axis=1
        )
        right_peer_action_valid = jnp.any(
            left_mask[:, lp:, lp:][:, :, -action_horizon:], axis=1
        )
        left_remote_mask = jnp.broadcast_to(
            left_peer_action_valid[:, None],
            (left_peer_action_valid.shape[0], left_suffix.shape[1], action_horizon),
        )
        right_remote_mask = jnp.broadcast_to(
            right_peer_action_valid[:, None],
            (right_peer_action_valid.shape[0], right_suffix.shape[1], action_horizon),
        )
        left_action_residual = _residual_action_update(
            left_qs,
            left_remote_action_k,
            left_remote_action_v,
            left_remote_mask,
            left_residual_params["kernel"],
            left_residual_params["gate"],
            configs[0],
        )
        right_action_residual = _residual_action_update(
            right_qs,
            right_remote_action_k,
            right_remote_action_v,
            right_remote_mask,
            right_residual_params["kernel"],
            right_residual_params["gate"],
            configs[0],
        )

    left_encoded_suffix = _attention(left_qs, left_suffix_k, left_suffix_v, left_suffix_mask, configs[0])
    right_encoded_suffix = _attention(right_qs, right_suffix_k, right_suffix_v, right_suffix_mask, configs[0])
    left_attn_prefix = _output_projection(left_encoded_prefix, left_params["attn"], configs[0], False)
    right_attn_prefix = _output_projection(right_encoded_prefix, right_params["attn"], configs[0], False)
    left_attn_prefix, right_attn_prefix = _fuse_common(left_attn_prefix, right_attn_prefix, ownership, common_enabled)
    left_attn_suffix = _output_projection(left_encoded_suffix, left_params["attn"], configs[1], True)
    right_attn_suffix = _output_projection(right_encoded_suffix, right_params["attn"], configs[1], True)

    left_prefix = left_prefix + left_attn_prefix
    right_prefix = right_prefix + right_attn_prefix
    left_suffix = left_suffix + left_attn_suffix * left_gate_suffix
    right_suffix = right_suffix + right_attn_suffix * right_gate_suffix
    if remote_action_residual_enabled:
        left_suffix = left_suffix + left_action_residual
        right_suffix = right_suffix + right_action_residual
    left_ff_prefix, _ = _rms_norm(left_prefix, left_params["pre_ffw_norm"], None)
    right_ff_prefix, _ = _rms_norm(right_prefix, right_params["pre_ffw_norm"], None)
    left_ff_prefix, right_ff_prefix = _fuse_common(left_ff_prefix, right_ff_prefix, ownership, common_enabled)
    left_ff_suffix, left_ff_gate = _rms_norm(left_suffix, left_params["pre_ffw_norm_1"], left_cond[1])
    right_ff_suffix, right_ff_gate = _rms_norm(right_suffix, right_params["pre_ffw_norm_1"], right_cond[1])
    left_ff_prefix = _feed_forward(left_ff_prefix, left_params["mlp"], configs[0])
    right_ff_prefix = _feed_forward(right_ff_prefix, right_params["mlp"], configs[0])
    left_ff_prefix, right_ff_prefix = _fuse_common(left_ff_prefix, right_ff_prefix, ownership, common_enabled)
    left_ff_suffix = _feed_forward(left_ff_suffix, left_params["mlp_1"], configs[1])
    right_ff_suffix = _feed_forward(right_ff_suffix, right_params["mlp_1"], configs[1])
    return (
        (left_prefix + left_ff_prefix, left_suffix + left_ff_suffix * left_ff_gate),
        (right_prefix + right_ff_prefix, right_suffix + right_ff_suffix * right_ff_gate),
    )


def paired_gemma_forward(
    left_params: dict[str, Any],
    right_params: dict[str, Any],
    configs: Sequence[Any],
    left_embedded,
    right_embedded,
    *,
    left_positions,
    right_positions,
    left_mask,
    right_mask,
    left_adarms_cond,
    right_adarms_cond,
    ownership,
    common_enabled: bool,
    private_kv_enabled: bool,
    remote_action_kv_enabled: bool = False,
    action_horizon: int | None = None,
    remote_action_residual_enabled: bool = False,
    left_residual_params: dict[str, Any] | None = None,
    right_residual_params: dict[str, Any] | None = None,
):
    """Execute two parameter-independent Gemma stacks with explicit communication."""

    depth = configs[0].depth
    if remote_action_kv_enabled and action_horizon is None:
        raise ValueError("action_horizon is required for Remote Action K/V")
    if remote_action_residual_enabled and action_horizon is None:
        raise ValueError("action_horizon is required for residual Remote Action")
    if remote_action_residual_enabled and (
        left_residual_params is None or right_residual_params is None
    ):
        raise ValueError("residual Remote Action requires receiver-local parameters")
    owner = _batched_ownership(
        ownership,
        batch_size=left_embedded[0].shape[0],
        token_count=left_embedded[0].shape[1],
    )
    left_xs, right_xs = tuple(x.astype(jnp.bfloat16) for x in left_embedded), tuple(x.astype(jnp.bfloat16) for x in right_embedded)

    def scan_layer(carry, layer_params):
        current_left, current_right = carry
        if remote_action_residual_enabled:
            layer_left, layer_right, layer_left_residual, layer_right_residual = layer_params
        else:
            layer_left, layer_right = layer_params
            layer_left_residual = layer_right_residual = None
        next_pair = _run_layer_pair(
            current_left,
            current_right,
            layer_left,
            layer_right,
            configs,
            left_positions,
            right_positions,
            left_mask,
            right_mask,
            left_adarms_cond,
            right_adarms_cond,
            ownership,
            common_enabled,
            private_kv_enabled,
            remote_action_kv_enabled,
            0 if action_horizon is None else action_horizon,
            remote_action_residual_enabled,
            layer_left_residual,
            layer_right_residual,
        )
        return next_pair, None
    scan_layer = jax.checkpoint(scan_layer, prevent_cse=False)

    if remote_action_residual_enabled:
        scan_inputs = (
            left_params["layers"],
            right_params["layers"],
            left_residual_params,
            right_residual_params,
        )
    else:
        scan_inputs = (left_params["layers"], right_params["layers"])

    (left_xs, right_xs), _ = jax.lax.scan(
        scan_layer,
        (left_xs, right_xs),
        scan_inputs,
        length=depth,
    )

    left_out = (
        _rms_norm(left_xs[0], left_params["final_norm"], None)[0],
        _rms_norm(left_xs[1], left_params["final_norm_1"], left_adarms_cond[1])[0],
    )
    right_out = (
        _rms_norm(right_xs[0], right_params["final_norm"], None)[0],
        _rms_norm(right_xs[1], right_params["final_norm_1"], right_adarms_cond[1])[0],
    )
    events = jnp.full(
        (depth, len(RAW_FUSION_POINTS)),
        1 if common_enabled else 0,
        dtype=jnp.int32,
    )
    trace = PairedGemmaTrace(
        events,
        jnp.max(jnp.sum(owner == PRIVATE, axis=-1)) if private_kv_enabled else 0,
        0 if not remote_action_kv_enabled else int(action_horizon),
        0 if not remote_action_residual_enabled else int(action_horizon),
    )
    return left_out, right_out, trace
