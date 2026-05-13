from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import pytest
from farm_edge_agent.safety.watchdog import Watchdog


class FakeClock:
    """Drives both `clock` and `sleep` so tests are deterministic.

    `sleep(d)` records the request and yields to the loop. Tests call
    `advance(seconds)` to bump the clock and wake any pending sleeper whose
    deadline has been reached.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self._waiter: asyncio.Future[None] | None = None
        self._deadline = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, d: float) -> Awaitable[None]:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._waiter = fut
        self._deadline = self.t + d
        return fut

    async def advance(self, seconds: float) -> None:
        self.t += seconds
        if (
            self._waiter is not None
            and self.t >= self._deadline
            and not self._waiter.done()
        ):
            self._waiter.set_result(None)
            self._waiter = None
        await asyncio.sleep(0)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_silence_fires_callback_at_timeout() -> None:
    fired = asyncio.Event()
    clock = FakeClock()

    def on_silence() -> None:
        fired.set()

    async with Watchdog(
        timeout_ms=1000, on_silence=on_silence, clock=clock.now, sleep=clock.sleep
    ):
        await clock.advance(0.999)
        assert not fired.is_set()
        await clock.advance(0.001)
        assert fired.is_set()


@pytest.mark.asyncio
async def test_kick_resets_timer() -> None:
    fired = asyncio.Event()
    clock = FakeClock()

    async with Watchdog(
        timeout_ms=1000,
        on_silence=lambda: fired.set(),
        clock=clock.now,
        sleep=clock.sleep,
    ) as wd:
        await clock.advance(0.9)
        wd.kick()
        await clock.advance(0.5)
        assert not fired.is_set()
        await clock.advance(0.5)
        assert fired.is_set()


@pytest.mark.asyncio
async def test_async_callback_awaited() -> None:
    seen: list[str] = []
    clock = FakeClock()

    async def on_silence() -> None:
        seen.append("awaited")

    async with Watchdog(
        timeout_ms=100,
        on_silence=on_silence,
        clock=clock.now,
        sleep=clock.sleep,
    ) as wd:
        await clock.advance(0.1)
        assert wd.fired
    assert seen == ["awaited"]


def test_invalid_timeout_rejected() -> None:
    try:
        Watchdog(timeout_ms=0, on_silence=lambda: None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
