"""In-process event bus used to fan out RunLoop events to SSE subscribers.

Each SSE client gets an asyncio.Queue; the publisher (a worker thread
running the RunLoop) drops events into every queue in a thread-safe way.
Disconnected subscribers are pruned lazily when a publish encounters a
full queue (treated as a dead reader).
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any


class EventBus:
    """Multi-subscriber broadcast with a small replay buffer per topic.

    Topics are arbitrary strings — typical names are ``run:<id>`` and
    ``world``. ``recent(topic)`` returns the buffered tail so a new
    subscriber doesn't miss events that fired before they connected.
    """

    def __init__(self, *, history: int = 200) -> None:
        self._lock = threading.Lock()
        self._subs: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._history_size = int(history)
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Thread-safe publish. Buffers into history; queues onto each
        subscriber's asyncio.Queue via call_soon_threadsafe."""
        with self._lock:
            hist = self._history.setdefault(topic, deque(maxlen=self._history_size))
            hist.append(event)
            subs = list(self._subs.get(topic, []))
        loop = self._loop
        if loop is None:
            return
        for q in subs:
            loop.call_soon_threadsafe(_put_nowait, q, event)

    async def subscribe(self, topic: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        with self._lock:
            self._subs.setdefault(topic, []).append(q)
        return q

    def unsubscribe(self, topic: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            try:
                self._subs.get(topic, []).remove(q)
            except ValueError:
                pass

    def recent(self, topic: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history.get(topic, []))


def _put_nowait(q: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        # Drop oldest — slow consumer
        try:
            q.get_nowait()
            q.put_nowait(event)
        except Exception:
            pass


__all__ = ["EventBus"]
