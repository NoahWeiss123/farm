from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from farm_edge_agent.errors import FarmError
from farm_edge_agent.protocol.client import WSClient
from farm_edge_agent.protocol.messages import (
    Ack,
    ActionChunk,
    Control,
    Hello,
    ObsChunk,
    TcpPose,
)
from farm_shared.errors import ErrorCode
from farm_shared.protocol import CURRENT_PROTOCOL
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


async def _accept_hello(ws: ServerConnection) -> Hello:
    raw = await ws.recv()
    hello = Hello.model_validate_json(raw)
    ack = Ack(protocol_version=str(CURRENT_PROTOCOL), accepted=True)
    await ws.send(ack.model_dump_json())
    return hello


def _make_obs(run_id: str = "r_0") -> ObsChunk:
    return ObsChunk(
        run_id=run_id,
        ts=1.0,
        frames={"wrist": "r2://run/wrist/0.jpg"},
        joint_state=[0.0] * 6,
        tcp_pose=TcpPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),
        gripper_state="open",
    )


def test_connect_does_handshake() -> None:
    seen: dict[str, Any] = {}

    async def handler(ws: ServerConnection) -> None:
        hello = await _accept_hello(ws)
        seen["agent_version"] = hello.agent_version
        seen["headers"] = dict(ws.request.headers)  # type: ignore[union-attr]
        await asyncio.sleep(0.5)

    async def go() -> None:
        async with _running(handler) as url:
            client = WSClient(agent_version="0.4.2")
            ack = await client.connect(url, "sk-test")
            assert ack.accepted is True
            assert ack.protocol_version == str(CURRENT_PROTOCOL)
            await client.close()

    _run(go())
    assert seen["agent_version"] == "0.4.2"
    headers_lower = {k.lower(): v for k, v in seen["headers"].items()}
    assert headers_lower.get("authorization") == "Bearer sk-test"


def test_connect_raises_on_protocol_mismatch() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.recv()
        ack = Ack(protocol_version="99.0.0", accepted=False, reason="too old")
        await ws.send(ack.model_dump_json())

    async def go() -> None:
        async with _running(handler) as url:
            client = WSClient(agent_version="0.0.1")
            with pytest.raises(FarmError) as ei:
                await client.connect(url, "sk-test")
            assert ei.value.code is ErrorCode.E1006
            assert client.connected is False

    _run(go())


def test_send_obs_arrives_at_server() -> None:
    received: list[ObsChunk] = []

    async def handler(ws: ServerConnection) -> None:
        await _accept_hello(ws)
        raw = await ws.recv()
        received.append(ObsChunk.model_validate_json(raw))
        await asyncio.sleep(0.2)

    async def go() -> None:
        async with _running(handler) as url:
            client = WSClient(agent_version="0.0.1")
            await client.connect(url, "sk-test")
            await client.send_obs(_make_obs("r_send"))
            await client.close()

    _run(go())
    assert len(received) == 1
    assert received[0].run_id == "r_send"


def test_iter_actions_yields_action_chunks() -> None:
    async def handler(ws: ServerConnection) -> None:
        await _accept_hello(ws)
        for i in range(3):
            chunk = ActionChunk(
                run_id="r_a", chunk_id=i, actions=[], suggested_dwell_ms=100
            )
            await ws.send(chunk.model_dump_json())
        await asyncio.sleep(0.5)

    chunks: list[ActionChunk] = []

    async def go() -> None:
        async with _running(handler) as url:
            client = WSClient(agent_version="0.0.1", watchdog_timeout_s=2.0)
            await client.connect(url, "sk-test")
            async for chunk in client.iter_actions():
                chunks.append(chunk)
                if len(chunks) == 3:
                    break
            await client.close()

    _run(go())
    assert [c.chunk_id for c in chunks] == [0, 1, 2]


def test_iter_actions_routes_control_to_handler() -> None:
    async def handler(ws: ServerConnection) -> None:
        await _accept_hello(ws)
        pause = Control(run_id="r_c", command="pause")
        await ws.send(pause.model_dump_json())
        chunk = ActionChunk(run_id="r_c", chunk_id=0, actions=[], suggested_dwell_ms=50)
        await ws.send(chunk.model_dump_json())
        await asyncio.sleep(0.5)

    seen_controls: list[Control] = []

    async def on_control(c: Control) -> None:
        seen_controls.append(c)

    chunks: list[ActionChunk] = []

    async def go() -> None:
        async with _running(handler) as url:
            client = WSClient(agent_version="0.0.1", watchdog_timeout_s=2.0)
            client.on_control(on_control)
            await client.connect(url, "sk-test")
            async for chunk in client.iter_actions():
                chunks.append(chunk)
                break
            await client.close()

    _run(go())
    assert len(seen_controls) == 1
    assert seen_controls[0].command == "pause"
    assert chunks[0].chunk_id == 0


def test_reconnects_on_transient_drop() -> None:
    state = {"connections": 0}

    async def handler(ws: ServerConnection) -> None:
        state["connections"] += 1
        n = state["connections"]
        await _accept_hello(ws)
        if n == 1:
            chunk = ActionChunk(
                run_id="r_r", chunk_id=0, actions=[], suggested_dwell_ms=50
            )
            await ws.send(chunk.model_dump_json())
            await ws.close()
        else:
            chunk = ActionChunk(
                run_id="r_r", chunk_id=1, actions=[], suggested_dwell_ms=50
            )
            await ws.send(chunk.model_dump_json())
            await asyncio.sleep(0.5)

    chunks: list[ActionChunk] = []

    async def go() -> None:
        async with _running(handler) as url:
            client = WSClient(
                agent_version="0.0.1",
                watchdog_timeout_s=2.0,
                reconnect_backoff_s=0.0,
            )
            await client.connect(url, "sk-test")
            async for chunk in client.iter_actions():
                chunks.append(chunk)
                if len(chunks) == 2:
                    break
            await client.close()

    _run(go())
    assert state["connections"] >= 2
    assert [c.chunk_id for c in chunks] == [0, 1]


def test_halts_on_server_silence() -> None:
    async def handler(ws: ServerConnection) -> None:
        await _accept_hello(ws)
        await asyncio.sleep(1.0)  # never sends an action

    async def go() -> None:
        async with _running(handler) as url:
            client = WSClient(agent_version="0.0.1", watchdog_timeout_s=0.15)
            await client.connect(url, "sk-test")
            with pytest.raises(FarmError) as ei:
                async for _chunk in client.iter_actions():
                    pass
            assert ei.value.code is ErrorCode.E3002
            await client.close()

    _run(go())


def test_drop_without_reachable_server_raises_e1005() -> None:
    async def handler(ws: ServerConnection) -> None:
        await _accept_hello(ws)
        await ws.close()  # drop the connection right after the handshake

    async def go() -> None:
        server = await serve(handler, "127.0.0.1", 0)
        sockets = server.sockets  # type: ignore[attr-defined]
        port = sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        client = WSClient(
            agent_version="0.0.1",
            watchdog_timeout_s=2.0,
            reconnect_backoff_s=0.0,
            reconnect_max_attempts=2,
        )
        await client.connect(url, "sk-test")

        async def consume() -> None:
            async for _ in client.iter_actions():
                pass

        consume_task = asyncio.create_task(consume())
        # Pull the server out from under the client so reconnect attempts fail.
        await asyncio.sleep(0.1)
        server.close()
        await server.wait_closed()
        with pytest.raises(FarmError) as ei:
            await consume_task
        assert ei.value.code is ErrorCode.E1005
        await client.close()

    _run(go())
