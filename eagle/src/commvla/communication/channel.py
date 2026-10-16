"""Instrumented two-rank communication channel for CommVLA inference."""

from __future__ import annotations

import csv
import gzip
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist


MessageGroup = Literal["action", "common", "remote_kv", "system"]
CodecName = Literal[
    "raw",
    "int8",
    "int8_packed",
    "fp8_e4m3",
    "fp8_e5m2",
    "int4",
    "mixed_i4_a",
    "mixed_i4_b",
    "delta_int8",
    "delta_int4",
]
ProfileMode = Literal["none", "bytes", "timing"]


def _parse_period(value: str | int) -> int:
    if isinstance(value, int):
        period = value
    elif value == "initial":
        period = 0
    else:
        period = int(value)
    if period < 0:
        raise ValueError(f"refresh period must be non-negative, got {value}")
    return period


@dataclass(frozen=True)
class ChannelConfig:
    action_refresh: int = 1
    common_refresh: int = 1
    remote_kv_refresh: int = 1
    action_codec: CodecName = "raw"
    common_codec: CodecName = "raw"
    remote_kv_codec: CodecName = "raw"
    delta_keyframe: int = 4
    profile_mode: ProfileMode = "none"

    @classmethod
    def from_values(
        cls,
        *,
        action_refresh: str | int = 1,
        common_refresh: str | int = 1,
        remote_kv_refresh: str | int = 1,
        action_codec: CodecName = "raw",
        common_codec: CodecName = "raw",
        remote_kv_codec: CodecName = "raw",
        delta_keyframe: int = 4,
        profile_mode: ProfileMode = "none",
    ) -> "ChannelConfig":
        if int(delta_keyframe) < 1:
            raise ValueError(f"delta_keyframe must be positive, got {delta_keyframe}")
        return cls(
            action_refresh=_parse_period(action_refresh),
            common_refresh=_parse_period(common_refresh),
            remote_kv_refresh=_parse_period(remote_kv_refresh),
            action_codec=action_codec,
            common_codec=common_codec,
            remote_kv_codec=remote_kv_codec,
            delta_keyframe=int(delta_keyframe),
            profile_mode=profile_mode,
        )


_FIELDNAMES = [
    "episode_id",
    "planning_round",
    "rank",
    "direction",
    "message_group",
    "message_type",
    "layer_id",
    "shape",
    "dtype",
    "raw_bytes",
    "transmitted_bytes",
    "metadata_bytes",
    "collective_calls",
    "fresh",
    "cache_age",
    "codec",
    "effective_codec",
    "keyframe",
    "delta_mean_abs",
    "delta_max_abs",
    "delta_near_zero_fraction",
    "prepare_seconds",
    "encode_seconds",
    "d2h_seconds",
    "transfer_seconds",
    "h2d_seconds",
    "decode_seconds",
    "total_seconds",
]


class _TraceWriter:
    def __init__(self, output_dir: Path | None, rank: int, mode: ProfileMode) -> None:
        self.mode = mode
        self._handle = None
        self._writer = None
        if output_dir is None or mode == "none":
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        self._handle = gzip.open(output_dir / f"rank{rank}_communication.csv.gz", "wt", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=_FIELDNAMES)
        self._writer.writeheader()

    def write(self, row: dict) -> None:
        if self._writer is not None:
            self._writer.writerow(row)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None


class PeerCommunicationChannel:
    """Exchange peer tensors while applying refresh and compression policies.

    Each process writes one row per logical one-way message. Summing rank 0 and
    rank 1 rows therefore gives bidirectional communication volume.
    """

    def __init__(self, config: ChannelConfig, output_dir: Path | None = None) -> None:
        if not dist.is_initialized() or dist.get_world_size() != 2:
            raise RuntimeError("PeerCommunicationChannel requires an initialized two-rank process group")
        self.config = config
        self.rank = dist.get_rank()
        self.backend = dist.get_backend()
        self.writer = _TraceWriter(output_dir, self.rank, config.profile_mode)
        self.episode_id = -1
        self.planning_round = -1
        self._cache: dict[tuple[str, int], tuple[torch.Tensor, int]] = {}
        self._local_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._totals: dict[str, float] = defaultdict(float)
        self._by_group: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._by_episode: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def begin_round(self, episode_id: int, planning_round: int) -> None:
        if int(episode_id) != self.episode_id:
            self.episode_id = int(episode_id)
            self._cache.clear()
            self._local_cache.clear()
        self.planning_round = int(planning_round)

    def _refresh_period(self, group: MessageGroup) -> int:
        if group == "action":
            return self.config.action_refresh
        if group == "common":
            return self.config.common_refresh
        if group == "remote_kv":
            return self.config.remote_kv_refresh
        return 1

    def _codec(self, group: MessageGroup, tensor: torch.Tensor, message_type: str) -> CodecName:
        if not tensor.is_floating_point():
            return "raw"
        if group == "action":
            return self.config.action_codec
        if group == "common":
            codec = self.config.common_codec
            if codec == "mixed_i4_a":
                if message_type in {"common_attention_output", "common_mlp_output"}:
                    return "int4"
                return "int8_packed"
            if codec == "mixed_i4_b":
                if message_type in {
                    "common_initial_hidden",
                    "common_norm",
                    "common_attention_output",
                    "common_mlp_norm",
                    "common_mlp_output",
                    "common_hidden",
                }:
                    return "int4"
                return "int8_packed"
            return codec
        if group == "remote_kv":
            return self.config.remote_kv_codec
        return "raw"

    def _should_refresh(self, key: tuple[str, int], group: MessageGroup) -> bool:
        if key not in self._cache:
            return True
        period = self._refresh_period(group)
        if period == 0:
            return False
        return self.planning_round % period == 0

    def _sync_cuda(self, tensor: torch.Tensor) -> None:
        if self.config.profile_mode == "timing" and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def _all_gather_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = [torch.empty_like(tensor) for _ in range(2)]
        dist.all_gather(gathered, tensor)
        return gathered[1 - self.rank]

    def _all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.backend == "gloo":
            return self._all_gather_cpu(tensor)
        gathered = [torch.empty_like(tensor) for _ in range(2)]
        dist.all_gather(gathered, tensor)
        return gathered[1 - self.rank]

    @staticmethod
    def _encode_byte_payload(value: torch.Tensor, codec: str) -> tuple[torch.Tensor, torch.Tensor]:
        value_f = value.float()
        if codec in {"int8_packed", "delta_int8"}:
            max_abs = value_f.abs().amax().clamp_min(1e-8)
            scale = (max_abs / 127.0).reshape(1)
            encoded = torch.round(value_f / scale).clamp(-127, 127).to(torch.int8)
            return encoded.contiguous().view(torch.uint8).reshape(-1), scale
        if codec == "fp8_e4m3":
            fp8_dtype = torch.float8_e4m3fn
            fp8_max = 448.0
        elif codec == "fp8_e5m2":
            fp8_dtype = torch.float8_e5m2
            fp8_max = 57344.0
        elif codec in {"int4", "delta_int4"}:
            max_abs = value_f.abs().amax().clamp_min(1e-8)
            scale = (max_abs / 7.0).reshape(1)
            quantized = torch.round(value_f / scale).clamp(-7, 7).to(torch.int16).reshape(-1)
            unsigned = (quantized + 8).to(torch.uint8)
            if unsigned.numel() % 2:
                unsigned = torch.cat([unsigned, torch.full((1,), 8, dtype=torch.uint8, device=unsigned.device)])
            packed = unsigned[0::2] | (unsigned[1::2] << 4)
            return packed.contiguous(), scale
        else:
            raise ValueError(f"Unsupported packed codec: {codec}")
        max_abs = value_f.abs().amax().clamp_min(1e-8)
        scale = (max_abs / fp8_max).reshape(1)
        encoded = (value_f / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return encoded.contiguous().view(torch.uint8).reshape(-1), scale

    @staticmethod
    def _decode_byte_payload(
        payload: torch.Tensor,
        scale: torch.Tensor,
        codec: str,
        *,
        numel: int,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if codec in {"int8_packed", "delta_int8"}:
            decoded = payload.view(torch.int8).float() * scale.float()
        elif codec == "fp8_e4m3":
            decoded = payload.view(torch.float8_e4m3fn).float() * scale.float()
        elif codec == "fp8_e5m2":
            decoded = payload.view(torch.float8_e5m2).float() * scale.float()
        elif codec in {"int4", "delta_int4"}:
            low = (payload & 0x0F).to(torch.int16) - 8
            high = ((payload >> 4) & 0x0F).to(torch.int16) - 8
            decoded = torch.stack([low, high], dim=1).reshape(-1)[:numel].float() * scale.float()
        else:
            raise ValueError(f"Unsupported packed codec: {codec}")
        return decoded.reshape(shape).to(dtype=dtype)

    def exchange(
        self,
        tensor: torch.Tensor,
        *,
        group: MessageGroup,
        message_type: str,
        layer_id: int = -1,
    ) -> torch.Tensor:
        key = (message_type, int(layer_id))
        raw_bytes = int(tensor.numel() * tensor.element_size())
        codec = self._codec(group, tensor, message_type)
        fresh = self._should_refresh(key, group)
        cache_age = 0
        prepare_seconds = encode_seconds = d2h_seconds = 0.0
        transfer_seconds = h2d_seconds = decode_seconds = 0.0
        transmitted_bytes = metadata_bytes = 0
        collective_calls = 0
        effective_codec = codec
        keyframe = True
        delta_mean_abs = delta_max_abs = delta_near_zero_fraction = 0.0
        total_started = time.perf_counter()

        if fresh:
            prepare_started = time.perf_counter()
            local = tensor.detach().contiguous()
            self._sync_cuda(local)
            prepare_seconds = time.perf_counter() - prepare_started

            value_to_encode = local
            if codec in {"delta_int8", "delta_int4"}:
                keyframe = key not in self._local_cache or self.planning_round % self.config.delta_keyframe == 0
                if not keyframe:
                    value_to_encode = local - self._local_cache[key].to(device=local.device, dtype=local.dtype)
                    delta_f = value_to_encode.float()
                    delta_mean_abs = float(delta_f.abs().mean().item())
                    delta_max_abs = float(delta_f.abs().amax().item())
                    delta_near_zero_fraction = float((delta_f.abs() <= 1e-3).float().mean().item())
                self._local_cache[key] = local.detach()

            if codec == "int8":
                encode_started = time.perf_counter()
                max_abs = value_to_encode.float().abs().amax().clamp_min(1e-8)
                scale = (max_abs / 127.0).reshape(1)
                encoded = torch.round(value_to_encode.float() / scale).clamp(-127, 127).to(torch.int8)
                self._sync_cuda(local)
                encode_seconds = time.perf_counter() - encode_started
                transmitted_bytes = int(encoded.numel() * encoded.element_size())
                metadata_bytes = int(scale.numel() * scale.element_size())
                collective_calls = 2

                d2h_started = time.perf_counter()
                encoded_transport = encoded.cpu() if self.backend == "gloo" else encoded
                scale_transport = scale.cpu() if self.backend == "gloo" else scale
                d2h_seconds = time.perf_counter() - d2h_started

                transfer_started = time.perf_counter()
                peer_encoded = self._all_gather(encoded_transport)
                peer_scale = self._all_gather(scale_transport)
                transfer_seconds = time.perf_counter() - transfer_started

                h2d_started = time.perf_counter()
                peer_encoded = peer_encoded.to(device=tensor.device)
                peer_scale = peer_scale.to(device=tensor.device)
                self._sync_cuda(tensor)
                h2d_seconds = time.perf_counter() - h2d_started

                decode_started = time.perf_counter()
                peer = (peer_encoded.float() * peer_scale.float()).to(dtype=tensor.dtype)
                self._sync_cuda(tensor)
                decode_seconds = time.perf_counter() - decode_started
            elif codec in {"int8_packed", "fp8_e4m3", "fp8_e5m2", "int4", "delta_int8", "delta_int4"}:
                encode_started = time.perf_counter()
                payload, scale = self._encode_byte_payload(value_to_encode, codec)
                scale_bytes = scale.contiguous().view(torch.uint8).reshape(-1)
                packet = torch.cat([scale_bytes, payload])
                self._sync_cuda(local)
                encode_seconds = time.perf_counter() - encode_started
                transmitted_bytes = int(payload.numel())
                metadata_bytes = int(scale_bytes.numel())
                collective_calls = 1

                d2h_started = time.perf_counter()
                packet_transport = packet.cpu() if self.backend == "gloo" else packet
                d2h_seconds = time.perf_counter() - d2h_started

                transfer_started = time.perf_counter()
                peer_packet = self._all_gather(packet_transport)
                transfer_seconds = time.perf_counter() - transfer_started

                h2d_started = time.perf_counter()
                peer_packet = peer_packet.to(device=tensor.device)
                self._sync_cuda(tensor)
                h2d_seconds = time.perf_counter() - h2d_started

                decode_started = time.perf_counter()
                peer_scale = peer_packet[:4].contiguous().view(torch.float32)
                peer_payload = peer_packet[4:].contiguous()
                peer_value = self._decode_byte_payload(
                    peer_payload,
                    peer_scale,
                    codec,
                    numel=tensor.numel(),
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                )
                if codec in {"delta_int8", "delta_int4"} and not keyframe:
                    previous_peer, _ = self._cache[key]
                    peer = previous_peer.to(device=tensor.device, dtype=tensor.dtype) + peer_value
                else:
                    peer = peer_value
                self._sync_cuda(tensor)
                decode_seconds = time.perf_counter() - decode_started
            else:
                transmitted_bytes = raw_bytes
                collective_calls = 1
                d2h_started = time.perf_counter()
                transport = local.cpu() if self.backend == "gloo" else local
                d2h_seconds = time.perf_counter() - d2h_started

                transfer_started = time.perf_counter()
                peer_transport = self._all_gather(transport)
                transfer_seconds = time.perf_counter() - transfer_started

                h2d_started = time.perf_counter()
                peer = peer_transport.to(device=tensor.device, dtype=tensor.dtype)
                self._sync_cuda(tensor)
                h2d_seconds = time.perf_counter() - h2d_started
            self._cache[key] = (peer.detach(), self.planning_round)
        else:
            peer, cached_round = self._cache[key]
            cache_age = self.planning_round - cached_round
            if peer.shape != tensor.shape:
                raise RuntimeError(
                    f"cached peer shape changed for {key}: cached={tuple(peer.shape)} current={tuple(tensor.shape)}"
                )
            peer = peer.to(device=tensor.device, dtype=tensor.dtype)

        total_seconds = time.perf_counter() - total_started
        row = {
                "episode_id": self.episode_id,
                "planning_round": self.planning_round,
                "rank": self.rank,
                "direction": f"{self.rank}->{1 - self.rank}",
                "message_group": group,
                "message_type": message_type,
                "layer_id": int(layer_id),
                "shape": "x".join(str(dim) for dim in tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "raw_bytes": raw_bytes,
                "transmitted_bytes": transmitted_bytes,
                "metadata_bytes": metadata_bytes,
                "collective_calls": collective_calls,
                "fresh": int(fresh),
                "cache_age": cache_age,
                "codec": codec,
                "effective_codec": effective_codec,
                "keyframe": int(keyframe),
                "delta_mean_abs": delta_mean_abs,
                "delta_max_abs": delta_max_abs,
                "delta_near_zero_fraction": delta_near_zero_fraction,
                "prepare_seconds": prepare_seconds,
                "encode_seconds": encode_seconds,
                "d2h_seconds": d2h_seconds,
                "transfer_seconds": transfer_seconds,
                "h2d_seconds": h2d_seconds,
                "decode_seconds": decode_seconds,
                "total_seconds": total_seconds,
            }
        self.writer.write(row)
        numeric = {
            "messages": 1,
            "fresh_messages": int(fresh),
            "stale_messages": 1 - int(fresh),
            "raw_bytes": raw_bytes,
            "transmitted_bytes": transmitted_bytes,
            "metadata_bytes": metadata_bytes,
            "collective_calls": collective_calls,
            "prepare_seconds": prepare_seconds,
            "encode_seconds": encode_seconds,
            "d2h_seconds": d2h_seconds,
            "transfer_seconds": transfer_seconds,
            "h2d_seconds": h2d_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
        }
        for field, value in numeric.items():
            self._totals[field] += value
            self._by_group[group][field] += value
            self._by_type[f"{group}/{message_type}"][field] += value
            self._by_episode[self.episode_id][field] += value
        return peer

    def summary(self) -> dict:
        return {
            "rank": self.rank,
            "config": {
                "action_refresh": self.config.action_refresh,
                "common_refresh": self.config.common_refresh,
                "remote_kv_refresh": self.config.remote_kv_refresh,
                "action_codec": self.config.action_codec,
                "common_codec": self.config.common_codec,
                "remote_kv_codec": self.config.remote_kv_codec,
                "delta_keyframe": self.config.delta_keyframe,
                "profile_mode": self.config.profile_mode,
            },
            "totals": dict(self._totals),
            "by_group": {key: dict(value) for key, value in self._by_group.items()},
            "by_type": {key: dict(value) for key, value in self._by_type.items()},
            "by_episode": {str(key): dict(value) for key, value in self._by_episode.items()},
        }

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


_PEER_CHANNEL: PeerCommunicationChannel | None = None


def set_peer_channel(channel: PeerCommunicationChannel | None) -> None:
    global _PEER_CHANNEL
    _PEER_CHANNEL = channel


def get_peer_channel() -> PeerCommunicationChannel | None:
    return _PEER_CHANNEL
