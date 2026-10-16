"""Communication channels and profiling utilities."""

from commvla.communication.channel import (
    ChannelConfig,
    PeerCommunicationChannel,
    get_peer_channel,
    set_peer_channel,
)

__all__ = ["ChannelConfig", "PeerCommunicationChannel", "get_peer_channel", "set_peer_channel"]
