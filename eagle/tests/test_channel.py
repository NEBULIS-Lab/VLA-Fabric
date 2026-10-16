from __future__ import annotations

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from commvla.communication import ChannelConfig, PeerCommunicationChannel
from commvla.evaluation import tabletop_two_process_core


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, port: int, output_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    channel = PeerCommunicationChannel(
        ChannelConfig.from_values(action_refresh=2, action_codec="int8", profile_mode="bytes"),
        Path(output_dir),
    )
    channel.begin_round(episode_id=7, planning_round=0)
    local0 = torch.tensor([1.0 + rank, -2.0 - rank], dtype=torch.float32)
    peer0 = channel.exchange(local0, group="action", message_type="action_token")
    expected0 = torch.tensor([2.0 - rank, -3.0 + rank], dtype=torch.float32)
    assert torch.allclose(peer0, expected0, atol=0.03)

    channel.begin_round(episode_id=7, planning_round=1)
    peer1 = channel.exchange(local0 + 10.0, group="action", message_type="action_token")
    assert torch.allclose(peer1, peer0)

    channel.begin_round(episode_id=7, planning_round=2)
    peer2 = channel.exchange(local0 + 10.0, group="action", message_type="action_token")
    assert torch.allclose(peer2, expected0 + 10.0, atol=0.1)
    channel.close()
    dist.destroy_process_group()


def test_two_rank_int8_transport_and_refresh(tmp_path: Path) -> None:
    mp.start_processes(
        _worker,
        args=(_free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    assert (tmp_path / "rank0_communication.csv.gz").exists()
    assert (tmp_path / "rank1_communication.csv.gz").exists()


def test_no_remote_private_context_skips_collective(monkeypatch) -> None:
    def fail_exchange(*_args, **_kwargs):
        raise AssertionError("no_remote must not call exchange_peer")

    monkeypatch.setattr(tabletop_two_process_core, "exchange_peer", fail_exchange)
    local_k = torch.randn(1, 2, 3, 4)
    local_v = torch.randn(1, 2, 3, 4)
    local_valid = torch.ones(1, 3)
    remote_k, remote_v, remote_valid = tabletop_two_process_core.private_remote_context(
        local_k,
        local_v,
        local_valid,
        mode="no_remote",
        layer_id=0,
    )
    assert torch.count_nonzero(remote_k) == 0
    assert torch.count_nonzero(remote_v) == 0
    assert torch.count_nonzero(remote_valid) == 0
