"""Async watchdog: halts the arm if upstream goes silent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType

OnSilence = Callable[[], Awaitable[None] | None]
Clock = Callable[[], float]


class Watchdog(AbstractAsyncContextManager["Watchdog"]):
    """Calls `on_silence` if `kick()` isn't invoked within `timeout_ms`.

    Designed for `async with Watchdog(...) as wd: ...`. A custom `clock` and
    `sleep` (defaulting to `loop.time` / `asyncio.sleep`) make this drivable
    by a fake clock in tests with no wall-clock dependency.
    """

    def __init__(
        self,
        timeout_ms: float,
        on_silence: OnSilence,
        *,
        clock: Clock | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        self._timeout_s = timeout_ms / 1000.0
        self._on_silence = on_silence
        self._clock = clock
        self._sleep = sleep or asyncio.sleep
        self._last_kick: float = 0.0
        self._task: asyncio.Task[None] | None = None
        self._fired = False

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_event_loop().time()

    def kick(self) -> None:
        self._last_kick = self._now()

    @property
    def fired(self) -> bool:
        return self._fired

    async def __aenter__(self) -> Watchdog:
        self._last_kick = self._now()
        self._fired = False
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        # Floating-point arithmetic on (t - last_kick) leaks ULPs of error.
        # Treat a remaining time within 1ns of zero as expired.
        epsilon = 1e-9
        try:
            while True:
                deadline = self._last_kick + self._timeout_s
                remaining = deadline - self._now()
                if remaining <= epsilon:
                    self._fired = True
                    result = self._on_silence()
                    if asyncio.iscoroutine(result):
                        await result
                    return
                await self._sleep(remaining)
        except asyncio.CancelledError:
            raise
