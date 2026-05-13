from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from farm_edge_agent.errors import FarmError
from farm_edge_agent.protocol.handshake import parse_protocol_version, perform_handshake
from farm_edge_agent.protocol.messages import Ack, Hello
from farm_shared.errors import ErrorCode
from farm_shared.protocol import CURRENT_PROTOCOL, ProtocolVersion
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve

Handler = Callable[[ServerConnection], Awaitable[None]]


@asynccontextmanager
async def _running(handler: Handler) -> Any:
    async with serve(handler, "127.0.0.1", 0) as server:
        sockets = server.sockets  # type: ignore[attr-defined]
        port = sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


def test_parse_protocol_version_round_trip() -> None:
    v = parse_protocol_version("1.2.3")
    assert v == ProtocolVersion(1, 2, 3)
    assert str(v) == "1.2.3"


def test_parse_protocol_version_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        parse_protocol_version("1.2")


def test_compatible_versions_succeed() -> None:
    async def handler(ws: ServerConnection) -> None:
        raw = await ws.recv()
        hello = Hello.model_validate_json(raw)
        # Server runs 1.5.0; client runs CURRENT_PROTOCOL (1.x.y) — compatible
        ack = Ack(protocol_version="1.5.0", accepted=True)
        await ws.send(ack.model_dump_json())
        assert hello.agent_version == "0.0.1"

    async def go() -> None:
        async with _running(handler) as url:
            async with connect(url) as ws:
                ack = await perform_handshake(ws, agent_version="0.0.1")
                assert ack.accepted is True
                assert ack.protocol_version == "1.5.0"

    _run(go())


def test_major_version_mismatch_raises_e1006() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.recv()
        # Server is on a different major
        ack = Ack(protocol_version="2.0.0", accepted=False, reason="major mismatch")
        await ws.send(ack.model_dump_json())

    async def go() -> None:
        async with _running(handler) as url:
            async with connect(url) as ws:
                with pytest.raises(FarmError) as ei:
                    await perform_handshake(ws, agent_version="0.0.1")
                assert ei.value.code is ErrorCode.E1006
                expected = (
                    f"[FARM-E1006] Edge Agent v{CURRENT_PROTOCOL} detected, "
                    "Dispatcher requires v2.0.0+"
                    " — fix: 'pip install -U farm-edge-agent'"
                )
                assert str(ei.value) == expected

    _run(go())


def test_accepted_but_incompatible_major_still_raises() -> None:
    # Defensive: if the server claims accepted=True but advertises an incompatible
    # major, the client must still refuse rather than proceed.
    async def handler(ws: ServerConnection) -> None:
        await ws.recv()
        ack = Ack(protocol_version="9.9.9", accepted=True)
        await ws.send(ack.model_dump_json())

    async def go() -> None:
        async with _running(handler) as url:
            async with connect(url) as ws:
                with pytest.raises(FarmError) as ei:
                    await perform_handshake(ws, agent_version="0.0.1")
                assert ei.value.code is ErrorCode.E1006

    _run(go())


def test_handshake_sends_feature_flags() -> None:
    seen: dict[str, bool] = {}

    async def handler(ws: ServerConnection) -> None:
        raw = await ws.recv()
        hello = Hello.model_validate_json(raw)
        seen.update(hello.feature_flags)
        ack = Ack(protocol_version=str(CURRENT_PROTOCOL), accepted=True)
        await ws.send(ack.model_dump_json())

    async def go() -> None:
        async with _running(handler) as url:
            async with connect(url) as ws:
                await perform_handshake(
                    ws,
                    agent_version="0.0.1",
                    feature_flags={"depth_camera": True, "ghosting": False},
                )

    _run(go())
    assert seen == {"depth_camera": True, "ghosting": False}
