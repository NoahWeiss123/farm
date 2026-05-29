"""Supervisor — owns one RobotBackend and broadcasts its state.

Backend-agnostic: same code path drives the MuJoCo sim and the real UF850.
The dashboard and ROS-TCP bridge both consume the same JSON snapshot.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from farm_edge_agent.backends.base import RobotBackend
from farm_edge_agent.server.bus import EventBus


class Supervisor:
    def __init__(self, bus: EventBus, *, backend: RobotBackend) -> None:
        self._bus = bus
        self._backend = backend
        self._backend.connect()
        # The ROS-TCP bridge sets this after construction so we can merge
        # the live Quest controller pose into every snapshot, regardless
        # of which backend is active (sim has no per-backend snapshot
        # mutation hook).
        self._bridge: Any | None = None
        # Same idea for the recorder — owned by the app and attached
        # here so snapshot() can advertise recording state to the
        # dashboard, and the bridge can drive start/cancel via the
        # Quest A/B buttons.
        self._recorder: Any | None = None
        self._stop = threading.Event()
        # Cached snapshot updated by the pump thread. Reads from
        # snapshot() are lock-free dict copies — no contention with the
        # sim's RLock from the HTTP event loop. Without this, the
        # dashboard's 30 Hz polling could starve the event loop
        # (every GET held the sim lock for a render-time and queued
        # behind the recorder thread).
        self._snap_cache: dict[str, Any] = {}
        self._world_thread = threading.Thread(target=self._world_pump, daemon=True)
        self._world_thread.start()

    def attach_bridge(self, bridge: Any) -> None:
        self._bridge = bridge

    def attach_recorder(self, recorder: Any) -> None:
        self._recorder = recorder

    @property
    def recorder(self) -> Any | None:
        return self._recorder

    @property
    def backend(self) -> RobotBackend:
        return self._backend

    def shutdown(self) -> None:
        self._stop.set()
        self._backend.disconnect()

    def snapshot(self) -> dict[str, Any]:
        # Fast path: return the cache. The pump thread refreshes it at
        # 60 Hz with the same enrichment. Callers that need ground-truth
        # *now* (the recorder worker) should use ``snapshot_live``.
        cached = self._snap_cache
        if cached:
            return dict(cached)
        return self.snapshot_live()

    def snapshot_live(self) -> dict[str, Any]:
        """Synchronously sample the backend + bridge + recorder. Used
        by the world pump and by callers that must observe the latest
        state (not the cache)."""
        snap = self._backend.snapshot()
        if self._bridge is not None:
            cp = getattr(self._bridge, "last_controller_pose", None)
            if cp is not None:
                snap["controller_pose"] = cp
            # Headset link state for the dashboard's Status panel.
            # "connected" = a Quest TCP client is attached;
            # "active"    = it's also pushing pose updates (within 2 s).
            # A stale TCP socket from a crashed Unity app reads as
            # connected-but-not-active so the user can see it.
            client_count = int(getattr(self._bridge, "client_count", 0))
            last_t = float(getattr(self._bridge, "_last_ctrl_t", 0.0))
            now = time.time()
            fresh = (now - last_t) < 2.0 if last_t > 0 else False
            snap["headset"] = {
                "connected": client_count > 0,
                "active": client_count > 0 and fresh,
                "clients": client_count,
                "last_pose_age_s": (now - last_t) if last_t > 0 else None,
            }
        # Drive-mode toggle so dashboard + headset HUD can show whether
        # right-trigger motion commits to the real arm or stays digital.
        snap["drive_real_arm"] = bool(getattr(self._backend, "drive_real_arm", False))
        if self._recorder is not None:
            try:
                snap["recording"] = self._recorder.state
            except Exception:
                pass
        return snap

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

    def set_ghost_target_pose(self, pose_mm_deg) -> dict[str, Any]:
        fn = getattr(self._backend, "set_ghost_target_pose", None)
        if fn is None:
            return {"error": "backend has no ghost target"}
        return fn(pose_mm_deg)

    def set_joint_target(self, joints_rad, *, gripper: float | None = None) -> dict[str, Any]:
        fn = getattr(self._backend, "set_joint_target", None)
        if fn is None:
            return {"error": "backend has no joint target"}
        return fn(joints_rad, gripper=gripper)

    def swap_cameras(self) -> dict[str, str]:
        swapper = getattr(self._backend, "swap_cameras", None)
        if swapper is None:
            return {}
        return swapper()

    def _world_pump(self) -> None:
        # State snapshot @60 Hz. No camera rendering here — dashboard
        # camera tiles are reserved for real hardware and are served
        # directly by backend.camera_jpeg in app.get_camera_jpeg.
        while not self._stop.is_set():
            try:
                snap = self.snapshot_live()
                self._snap_cache = snap
                self._bus.publish("world", {"type": "world_snapshot", **snap})
            except Exception:
                pass
            self._stop.wait(1.0 / 60.0)


# Back-compat alias — supervisor tests still import the old name.
SimSupervisor = Supervisor

__all__ = ["Supervisor", "SimSupervisor"]
