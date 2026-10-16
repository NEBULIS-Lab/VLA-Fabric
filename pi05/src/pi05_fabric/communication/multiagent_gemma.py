"""Agent-parallel PI0.5 Gemma execution for two-to-four complete agents."""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import einops
import jax
import jax.numpy as jnp

from pi05_fabric.communication.multiagent_collectives import fuse_common_group
from pi05_fabric.communication.multiagent_collectives import gather_agent_values
from pi05_fabric.communication.multiagent_collectives import select_role_ordered_peers
from pi05_fabric.communication.paired_gemma import _apply_rope
from pi05_fabric.communication.paired_gemma import _attention
from pi05_fabric.communication.paired_gemma import _batched_ownership
from pi05_fabric.communication.paired_gemma import _feed_forward
from pi05_fabric.communication.paired_gemma import _output_projection
from pi05_fabric.communication.paired_gemma import _project_qkv
from pi05_fabric.communication.paired_gemma import _rms_norm
from pi05_fabric.communication.paired_gemma import action_suffix_kv
from pi05_fabric.communication.raw_common import RAW_FUSION_POINTS
from pi05_fabric.data.aloha_dual_agent import COMMON
from pi05_fabric.data.aloha_dual_agent import PRIVATE


class MultiAgentGemmaTrace(NamedTuple):
    raw_event_counts: jax.Array
    remote_private_tokens: jax.Array
    residual_action_tokens: int
    peer_slots: int
    transmitted_queries: int = 0
    transmitted_raw_observations: int = 0


def residual_peer_contexts(
    query,
    peer_keys,
    peer_values,
    peer_valid,
    *,
    num_kv_heads: int,
):
    """Attend to each peer independently, then concatenate by sender role."""

    query = jnp.asarray(query)
    keys = jnp.asarray(peer_keys)
    values = jnp.asarray(peer_values)
    valid = jnp.asarray(peer_valid, dtype=jnp.bool_)
    if keys.shape != values.shape or keys.ndim != 5:
        raise ValueError("peer K/V must have shape [slots, batch, tokens, heads, dim]")
    if valid.shape != keys.shape[:3]:
        raise ValueError("peer validity must have shape [slots, batch, tokens]")
    if query.ndim != 4 or query.shape[0] != keys.shape[1]:
        raise ValueError("query must have shape [batch, tokens, heads, dim]")
    if query.shape[2] % num_kv_heads:
        raise ValueError("query heads must be divisible by K/V heads")

    grouped_query = einops.rearrange(
        query,
        "B T (K G) H -> B T K G H",
        K=num_kv_heads,
    )
    logits = jnp.einsum(
        "BTKGH,SBUKH->SBKGTU",
        grouped_query,
        keys,
        preferred_element_type=jnp.float32,
    )
    logits = jnp.where(valid[:, :, None, None, None, :], logits, -2.3819763e38)
    probabilities = jax.nn.softmax(logits, axis=-1).astype(values.dtype)
    encoded = jnp.einsum("SBKGTU,SBUKH->SBTKGH", probabilities, values)
    slot_is_valid = jnp.any(valid, axis=-1)
    encoded = encoded * slot_is_valid[:, :, None, None, None, None]
    encoded = einops.rearrange(encoded, "S B T K G H -> B T (S K G H)")
    return encoded


def _peer_slots(local_value, *, role, agent_count, max_agents, axis_name, groups):
    gathered = gather_agent_values(
        local_value,
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    return select_role_ordered_peers(
        gathered,
        receiver_role=role,
        agent_count=agent_count,
        max_agents=max_agents,
    )


def _run_local_layer(
    xs,
    layer_params,
    configs,
    positions,
    mask,
    cond,
    ownership,
    *,
    role,
    agent_count,
    max_agents,
    axis_name,
    agent_groups,
    common_enabled,
    private_kv_enabled,
    residual_action_enabled,
    action_horizon,
    residual_params,
):
    prefix, suffix = xs
    prefix_length = prefix.shape[1]
    owner = _batched_ownership(
        ownership,
        batch_size=prefix.shape[0],
        token_count=prefix_length,
    )

    def fuse(value):
        if not common_enabled:
            return value
        return fuse_common_group(
            value,
            owner,
            common_value=COMMON,
            axis_name=axis_name,
            axis_index_groups=agent_groups,
            agent_count=agent_count,
        )

    norm_prefix, _ = _rms_norm(prefix, layer_params["pre_attention_norm"], None)
    norm_prefix = fuse(norm_prefix)
    norm_suffix, suffix_gate = _rms_norm(
        suffix,
        layer_params["pre_attention_norm_1"],
        cond[1],
    )
    qp, kp, vp = _project_qkv(norm_prefix, layer_params["attn"], configs[0], False)
    qp, kp, vp = fuse(qp), fuse(kp), fuse(vp)
    qs, ks, vs = _project_qkv(norm_suffix, layer_params["attn"], configs[1], True)

    qp = _apply_rope(qp, positions[:, :prefix_length]) * configs[0].head_dim**-0.5
    qs = _apply_rope(qs, positions[:, prefix_length:]) * configs[0].head_dim**-0.5
    kp = _apply_rope(kp, positions[:, :prefix_length])
    ks = _apply_rope(ks, positions[:, prefix_length:])

    local_k = jnp.concatenate([kp, ks], axis=1)
    local_v = jnp.concatenate([vp, vs], axis=1)
    encoded_prefix = _attention(
        qp,
        local_k,
        local_v,
        mask[:, :prefix_length],
        configs[0],
    )

    suffix_k, suffix_v = local_k, local_v
    suffix_mask = mask[:, prefix_length:]
    remote_private_tokens = jnp.asarray(0, dtype=jnp.int32)
    if private_kv_enabled:
        peer_kp, peer_slots_valid = _peer_slots(
            kp,
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            groups=agent_groups,
        )
        peer_vp, _ = _peer_slots(
            vp,
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            groups=agent_groups,
        )
        local_prefix_valid = mask[:, prefix_length, :prefix_length]
        peer_prefix_valid, _ = _peer_slots(
            local_prefix_valid,
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            groups=agent_groups,
        )
        local_private = owner == PRIVATE
        peer_private, _ = _peer_slots(
            local_private,
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            groups=agent_groups,
        )
        peer_valid = jnp.logical_and(peer_prefix_valid, peer_private)
        peer_valid = jnp.logical_and(
            peer_valid,
            peer_slots_valid[:, None, None],
        )
        peer_kp = einops.rearrange(peer_kp, "S B P K H -> B (S P) K H")
        peer_vp = einops.rearrange(peer_vp, "S B P K H -> B (S P) K H")
        peer_valid = einops.rearrange(peer_valid, "S B P -> B (S P)")
        suffix_k = jnp.concatenate([suffix_k, peer_kp], axis=1)
        suffix_v = jnp.concatenate([suffix_v, peer_vp], axis=1)
        suffix_mask = jnp.concatenate(
            [
                suffix_mask,
                jnp.broadcast_to(
                    peer_valid[:, None],
                    (peer_valid.shape[0], suffix.shape[1], peer_valid.shape[1]),
                ),
            ],
            axis=-1,
        )
        remote_private_tokens = jnp.max(jnp.sum(peer_valid, axis=-1))

    action_residual = None
    if residual_action_enabled:
        peer_ks, peer_slots_valid = _peer_slots(
            action_suffix_kv(ks, vs, action_horizon=action_horizon)[0],
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            groups=agent_groups,
        )
        peer_vs, _ = _peer_slots(
            action_suffix_kv(ks, vs, action_horizon=action_horizon)[1],
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            groups=agent_groups,
        )
        action_valid = jnp.any(
            mask[:, prefix_length:, prefix_length:][:, :, -action_horizon:],
            axis=1,
        )
        peer_action_valid, _ = _peer_slots(
            action_valid,
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            groups=agent_groups,
        )
        peer_action_valid = jnp.logical_and(
            peer_action_valid,
            peer_slots_valid[:, None, None],
        )
        contexts = residual_peer_contexts(
            qs,
            peer_ks,
            peer_vs,
            peer_action_valid,
            num_kv_heads=configs[0].num_kv_heads,
        )
        action_residual = jnp.einsum(
            "BTD,DF->BTF",
            contexts,
            residual_params["kernel"].astype(contexts.dtype),
        ) * residual_params["gate"].astype(contexts.dtype)

    encoded_suffix = _attention(qs, suffix_k, suffix_v, suffix_mask, configs[0])
    attention_prefix = fuse(
        _output_projection(encoded_prefix, layer_params["attn"], configs[0], False)
    )
    attention_suffix = _output_projection(
        encoded_suffix,
        layer_params["attn"],
        configs[1],
        True,
    )
    prefix = prefix + attention_prefix
    suffix = suffix + attention_suffix * suffix_gate
    if action_residual is not None:
        suffix = suffix + action_residual

    ff_prefix, _ = _rms_norm(prefix, layer_params["pre_ffw_norm"], None)
    ff_prefix = fuse(ff_prefix)
    ff_suffix, ff_gate = _rms_norm(
        suffix,
        layer_params["pre_ffw_norm_1"],
        cond[1],
    )
    ff_prefix = fuse(_feed_forward(ff_prefix, layer_params["mlp"], configs[0]))
    ff_suffix = _feed_forward(ff_suffix, layer_params["mlp_1"], configs[1])
    return (
        prefix + ff_prefix,
        suffix + ff_suffix * ff_gate,
    ), remote_private_tokens


def multiagent_gemma_forward(
    local_params: dict[str, Any],
    configs: Sequence[Any],
    embedded,
    *,
    positions,
    mask,
    adarms_cond,
    ownership,
    agent_count: int,
    max_agents: int,
    axis_name: str,
    agent_groups,
    common_enabled: bool,
    private_kv_enabled: bool,
    residual_action_enabled: bool,
    action_horizon: int,
    residual_params: dict[str, Any] | None,
):
    """Run one local policy while communicating inside its agent group."""

    if not 2 <= agent_count <= max_agents:
        raise ValueError("agent_count must be between two and max_agents")
    if residual_action_enabled and residual_params is None:
        raise ValueError("residual Action requires receiver-local parameters")
    role = jax.lax.axis_index(axis_name) % agent_count
    owner = _batched_ownership(
        ownership,
        batch_size=embedded[0].shape[0],
        token_count=embedded[0].shape[1],
    )
    xs = tuple(value.astype(jnp.bfloat16) for value in embedded)

    def scan_layer(carry, layer_inputs):
        current_xs, max_private_tokens = carry
        if residual_action_enabled:
            layer_params, layer_residual = layer_inputs
        else:
            layer_params, layer_residual = layer_inputs, None
        next_xs, private_tokens = _run_local_layer(
            current_xs,
            layer_params,
            configs,
            positions,
            mask,
            adarms_cond,
            ownership,
            role=role,
            agent_count=agent_count,
            max_agents=max_agents,
            axis_name=axis_name,
            agent_groups=agent_groups,
            common_enabled=common_enabled,
            private_kv_enabled=private_kv_enabled,
            residual_action_enabled=residual_action_enabled,
            action_horizon=action_horizon,
            residual_params=layer_residual,
        )
        return (next_xs, jnp.maximum(max_private_tokens, private_tokens)), None

    scan_layer = jax.checkpoint(scan_layer, prevent_cse=False)
    scan_inputs = local_params["layers"]
    if residual_action_enabled:
        scan_inputs = (scan_inputs, residual_params)
    (xs, private_tokens), _ = jax.lax.scan(
        scan_layer,
        (xs, jnp.asarray(0, dtype=jnp.int32)),
        scan_inputs,
        length=configs[0].depth,
    )
    output = (
        _rms_norm(xs[0], local_params["final_norm"], None)[0],
        _rms_norm(xs[1], local_params["final_norm_1"], adarms_cond[1])[0],
    )
    trace = MultiAgentGemmaTrace(
        raw_event_counts=jnp.full(
            (configs[0].depth, len(RAW_FUSION_POINTS)),
            1 if common_enabled else 0,
            dtype=jnp.int32,
        ),
        remote_private_tokens=private_tokens,
        residual_action_tokens=(agent_count - 1) * action_horizon
        if residual_action_enabled
        else 0,
        peer_slots=max_agents - 1,
    )
    return output, trace
