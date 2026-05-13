from __future__ import annotations

from typing import Protocol

from farm_edge_agent.errors import FarmError
from farm_edge_agent.protocol.messages import Ack, Hello
from farm_shared.errors import ErrorCode
from farm_shared.protocol import CURRENT_PROTOCOL, ProtocolVersion


class _AsyncSocket(Protocol):
    async def send(self, data: str) -> None: ...
    async def recv(self) -> str | bytes: ...


def parse_protocol_version(s: str) -> ProtocolVersion:
    parts = s.split(".")
    if len(parts) != 3:
        raise ValueError(f"invalid protocol version: {s!r}")
    return ProtocolVersion(int(parts[0]), int(parts[1]), int(parts[2]))


async def perform_handshake(
    ws: _AsyncSocket,
    *,
    agent_version: str,
    feature_flags: dict[str, bool] | None = None,
    client_protocol: ProtocolVersion = CURRENT_PROTOCOL,
) -> Ack:
    """Send ``Hello``, await ``Ack``, raise ``FarmError(E1006)`` on mismatch.

    The mismatch case fires when the server returns ``accepted=False`` or
    when the server's advertised protocol major differs from ours, since
    only major-version equality is wire-compatible.
    """
    hello = Hello(
        protocol_version=str(client_protocol),
        agent_version=agent_version,
        feature_flags=feature_flags or {},
    )
    await ws.send(hello.model_dump_json())
    raw = await ws.recv()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    ack = Ack.model_validate_json(raw)
    server_protocol = parse_protocol_version(ack.protocol_version)
    if not ack.accepted or not client_protocol.is_compatible_with(server_protocol):
        raise FarmError(
            ErrorCode.E1006,
            agent_version=str(client_protocol),
            required_version=str(server_protocol),
        )
    return ack
