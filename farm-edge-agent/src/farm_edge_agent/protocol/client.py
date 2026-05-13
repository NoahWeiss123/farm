from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from farm_shared.errors import ErrorCode
from farm_shared.protocol import CURRENT_PROTOCOL, ProtocolVersion
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from farm_edge_agent.errors import FarmError
from farm_edge_agent.protocol.handshake import perform_handshake
from farm_edge_agent.protocol.messages import (
    Ack,
    ActionChunk,
    Control,
    ObsChunk,
    parse_message,
)

WATCHDOG_TIMEOUT_S: float = 1.0
RECONNECT_BACKOFF_S: float = 0.1
RECONNECT_MAX_ATTEMPTS: int = 3


ControlHandler = Callable[[Control], Awaitable[None]]


class WSClient:
    """Edge Agent ↔ Dispatcher WebSocket client.

    Performs the version-negotiated handshake on connect, streams observations
    to the dispatcher, and yields incoming action chunks. Reconnects on
    transient connection drops. Halts with :data:`ErrorCode.E3002` when the
    dispatcher has been silent longer than ``watchdog_timeout_s`` (1s by spec).
    """

    def __init__(
        self,
        *,
        agent_version: str,
        client_protocol: ProtocolVersion = CURRENT_PROTOCOL,
        watchdog_timeout_s: float = WATCHDOG_TIMEOUT_S,
        reconnect_max_attempts: int = RECONNECT_MAX_ATTEMPTS,
        reconnect_backoff_s: float = RECONNECT_BACKOFF_S,
    ) -> None:
        self._agent_version = agent_version
        self._client_protocol = client_protocol
        self._watchdog_timeout_s = watchdog_timeout_s
        self._reconnect_max_attempts = reconnect_max_attempts
        self._reconnect_backoff_s = reconnect_backoff_s
        self._ws: ClientConnection | None = None
        self._url: str | None = None
        self._api_key: str | None = None
        self._feature_flags: dict[str, bool] = {}
        self._control_handler: ControlHandler | None = None
        self._current_run_id: str = "?"

    def on_control(self, handler: ControlHandler) -> None:
        """Register a coroutine invoked for every ``Control`` message received."""
        self._control_handler = handler

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(
        self,
        url: str,
        api_key: str,
        feature_flags: dict[str, bool] | None = None,
    ) -> Ack:
        self._url = url
        self._api_key = api_key
        self._feature_flags = dict(feature_flags or {})
        return await self._open()

    async def _open(self) -> Ack:
        assert self._url is not None and self._api_key is not None
        headers = {"Authorization": f"Bearer {self._api_key}"}
        ws = await connect(self._url, additional_headers=headers)
        try:
            ack = await perform_handshake(
                ws,
                agent_version=self._agent_version,
                feature_flags=self._feature_flags,
                client_protocol=self._client_protocol,
            )
        except BaseException:
            await ws.close()
            raise
        self._ws = ws
        return ack

    async def send_obs(self, obs: ObsChunk) -> None:
        if self._ws is None:
            raise RuntimeError("WSClient not connected")
        self._current_run_id = obs.run_id
        await self._ws.send(obs.model_dump_json())

    async def iter_actions(self) -> AsyncIterator[ActionChunk]:
        """Yield ``ActionChunk`` messages; route ``Control`` messages to the handler.

        Raises :class:`FarmError` with :data:`ErrorCode.E3002` on watchdog timeout
        (>``watchdog_timeout_s`` of server silence) and :data:`ErrorCode.E1005`
        if the connection drops and reconnection fails.
        """
        while True:
            if self._ws is None:
                raise RuntimeError("WSClient not connected")
            try:
                raw = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self._watchdog_timeout_s,
                )
            except TimeoutError:
                raise FarmError(ErrorCode.E3002) from None
            except ConnectionClosed:
                if not await self._reconnect():
                    raise FarmError(
                        ErrorCode.E1005,
                        seconds=self._watchdog_timeout_s,
                        run_id=self._current_run_id,
                    ) from None
                continue

            msg = parse_message(raw)
            if isinstance(msg, ActionChunk):
                yield msg
            elif isinstance(msg, Control):
                if self._control_handler is not None:
                    await self._control_handler(msg)

    async def _reconnect(self) -> bool:
        self._ws = None
        for attempt in range(self._reconnect_max_attempts):
            try:
                await self._open()
                return True
            except (OSError, WebSocketException, FarmError):
                if attempt + 1 < self._reconnect_max_attempts:
                    await asyncio.sleep(self._reconnect_backoff_s)
        return False

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
