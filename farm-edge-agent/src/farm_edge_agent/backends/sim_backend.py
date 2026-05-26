"""SimBackend — adapt the existing MuJoCo ``Sim`` to ``RobotBackend``."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from farm_edge_agent.sim import Sim

from .base import GripperState, JogAxis


class SimBackend:
    backend_name = "sim"

    def __init__(self, sim: Sim | None = None) -> None:
        self._sim = sim or Sim()
        self._estopped = False

    def connect(self) -> None:
        self._sim.connect()

    def disconnect(self) -> None:
        self._sim.disconnect()

    def snapshot(self) -> dict[str, Any]:
        snap = self._sim.snapshot()
        snap.setdefault("t", time.time())
        snap["backend"] = self.backend_name
        snap["estopped"] = self._estopped
        snap["cameras"] = ["exterior", "wrist", "topdown"]
        # In the sim, kinematic move_to teleports the arm — desired and
        # actual coincide. Echo joints into target_joints so the
        # dashboard's ghost-arm code has a uniform field to consume.
        snap["target_joints"] = list(snap.get("joints", []))
        return snap

    @property
    def cameras(self) -> list[str]:
        return ["exterior", "wrist", "topdown"]

    def swap_cameras(self) -> dict[str, str]:
        return {}

    def render_rgb(self, camera: str, *, width: int, height: int) -> np.ndarray:
        return self._sim.render_rgb(camera=camera, height=height, width=width)

    def jog(
        self, axis: JogAxis, sign: int, *, step_mm: float, step_rad: float
    ) -> dict[str, Any]:
        if self._estopped:
            raise RuntimeError("sim backend is e-stopped; call estop_clear first")
        new_pose = self._sim.jog(axis, sign, step_mm=step_mm, step_rad=step_rad)
        return {"pose": list(new_pose), "snapshot": self.snapshot()}

    def home(self) -> dict[str, Any]:
        if self._estopped:
            raise RuntimeError("sim backend is e-stopped; call estop_clear first")
        self._sim.home()
        return self.snapshot()

    def set_gripper(self, state: GripperState) -> dict[str, Any]:
        if self._estopped:
            raise RuntimeError("sim backend is e-stopped; call estop_clear first")
        self._sim.set_gripper(state)
        return self.snapshot()

    def estop(self) -> dict[str, Any]:
        self._estopped = True
        return {"estopped": True}

    def estop_clear(self) -> dict[str, Any]:
        self._estopped = False
        return {"estopped": False}
