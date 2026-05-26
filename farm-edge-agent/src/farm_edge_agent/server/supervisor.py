"""Supervisor — owns one RobotBackend and broadcasts its state.

Backend-agnostic: same code path drives the MuJoCo sim and the real UF850.
The dashboard and ROS-TCP bridge both consume the same JSON snapshot.
"""

from __future__ import annotations

import threading
from typing import Any

from farm_edge_agent.backends.base import RobotBackend
from farm_edge_agent.server.bus import EventBus


class Supervisor:
    def __init__(self, bus: EventBus, *, backend: RobotBackend) -> None:
        self._bus = bus
        self._backend = backend
        self._backend.connect()
        self._stop = threading.Event()
        self._world_thread = threading.Thread(target=self._world_pump, daemon=True)
        self._world_thread.start()

    @property
    def backend(self) -> RobotBackend:
        return self._backend

    def shutdown(self) -> None:
        self._stop.set()
        self._backend.disconnect()

    def snapshot(self) -> dict[str, Any]:
        return self._backend.snapshot()

    def render_camera(self, name: str, *, width: int, height: int):
        return self._backend.render_rgb(name, width=width, height=height)

    def jog(self, axis: str, sign: int, *, step_mm: float, step_rad: float) -> dict[str, Any]:
        return self._backend.jog(axis, sign, step_mm=step_mm, step_rad=step_rad)  # type: ignore[arg-type]

    def home(self) -> dict[str, Any]:
        return self._backend.home()

    def set_gripper(self, state: str) -> dict[str, Any]:
        if state not in ("open", "closed"):
            raise ValueError(f"gripper state must be 'open' or 'closed', got {state!r}")
        return self._backend.set_gripper(state)  # type: ignore[arg-type]

    def estop(self) -> dict[str, Any]:
        return self._backend.estop()

    def estop_clear(self) -> dict[str, Any]:
        return self._backend.estop_clear()

    def cameras(self) -> list[str]:
        return list(getattr(self._backend, "cameras", []))

    def swap_cameras(self) -> dict[str, str]:
        swapper = getattr(self._backend, "swap_cameras", None)
        if swapper is None:
            return {}
        return swapper()

    def _world_pump(self) -> None:
        while not self._stop.is_set():
            try:
                snap = self._backend.snapshot()
                self._bus.publish("world", {"type": "world_snapshot", **snap})
            except Exception:
                pass
            self._stop.wait(0.2)


# Back-compat alias — supervisor tests still import the old name.
SimSupervisor = Supervisor

__all__ = ["Supervisor", "SimSupervisor"]
