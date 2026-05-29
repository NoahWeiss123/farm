"""SimBackend — adapt the existing MuJoCo ``Sim`` to ``RobotBackend``."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from farm_edge_agent.cameras import RealsenseGrabber, RealsenseUnavailable
from farm_edge_agent.sim import Sim

from .base import GripperState, JogAxis

log = logging.getLogger("farm.backends.sim")


class SimBackend:
    backend_name = "sim"

    def __init__(
        self,
        sim: Sim | None = None,
        *,
        cameras: bool = True,
        camera_mapping: dict[str, str] | None = None,
    ) -> None:
        self._sim = sim or Sim()
        self._estopped = False
        # Sim is the digital arm — "drive_real_arm" is a no-op here but
        # we keep the attribute for API symmetry with xarm_backend so
        # the bridge's right-stick-click toggle code is backend-agnostic.
        self.drive_real_arm = False
        # Rate-cap ghost-target updates from the Quest bridge so we
        # don't burn the IK solver at the full 30 Hz Quest frame rate.
        # The sim is kinematic — visually 60 Hz is plenty.
        self._ghost_last_t = 0.0
        self._ghost_min_dt = 1.0 / 60.0
        self._grabber: RealsenseGrabber | None = None
        if cameras:
            try:
                self._grabber = RealsenseGrabber(mapping=camera_mapping)
                log.info("realsense grabber initialized: %s", self._grabber.names())
            except RealsenseUnavailable as exc:
                log.warning("realsense cameras unavailable: %s", exc)

    def connect(self) -> None:
        self._sim.connect()
        if self._grabber is not None:
            self._grabber.start()

    def disconnect(self) -> None:
        if self._grabber is not None:
            try:
                self._grabber.stop()
            except Exception as exc:
                log.warning("grabber stop raised: %s", exc)
        self._sim.disconnect()

    def snapshot(self) -> dict[str, Any]:
        snap = self._sim.snapshot()
        snap.setdefault("t", time.time())
        snap["backend"] = self.backend_name
        snap["estopped"] = self._estopped
        snap["cameras"] = self.cameras
        # In the sim, kinematic move_to teleports the arm — desired and
        # actual coincide. Echo joints into target_joints so the
        # dashboard's ghost-arm code has a uniform field to consume.
        snap["target_joints"] = list(snap.get("joints", []))
        return snap

    @property
    def cameras(self) -> list[str]:
        if self._grabber is None:
            return []
        return self._grabber.names()

    def swap_cameras(self) -> dict[str, str]:
        if self._grabber is None:
            return {}
        return self._grabber.swap()

    def camera_jpeg(self, camera: str) -> bytes | None:
        if self._grabber is None:
            return None
        return self._grabber.latest_jpeg(camera)

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

    def set_ghost_target_pose(
        self, pose_mm_deg: tuple[float, float, float, float, float, float]
    ) -> dict[str, Any]:
        """Drive the sim TCP to the Quest-derived target.

        The sim is kinematic, so "ghost" and actual arm coincide — when
        the bridge tells us to go somewhere, we just IK + apply. The
        bridge sends mm + degrees (xArm convention); the underlying
        ``sim.move_to`` takes mm + radians, so we convert here.
        """
        if self._estopped:
            return {"error": "sim is e-stopped"}
        now = time.time()
        if now - self._ghost_last_t < self._ghost_min_dt:
            return {"throttled": True}
        self._ghost_last_t = now
        x, y, z, rx_deg, ry_deg, rz_deg = pose_mm_deg
        pose_mm_rad = (
            float(x), float(y), float(z),
            math.radians(float(rx_deg)),
            math.radians(float(ry_deg)),
            math.radians(float(rz_deg)),
        )
        try:
            self._sim.move_to(pose_mm_rad)
        except Exception as exc:
            return {"error": f"move_to failed: {exc}"}
        return {"ghost": "applied"}

    def set_joint_target(
        self,
        joints_rad: list[float] | tuple[float, ...],
        *,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        """Joint-space sibling of ``set_ghost_target_pose``. Kinematic
        sim → teleport directly. Gripper is thresholded at 0.5 onto the
        sim's binary ``set_gripper`` since the sim model only has two
        commanded positions."""
        if self._estopped:
            return {"error": "estopped"}
        joints = list(joints_rad)
        if len(joints) != 6:
            return {"error": f"joints must have length 6, got {len(joints)}"}
        try:
            self._sim.move_joint([float(j) for j in joints])
        except Exception as exc:
            return {"error": f"move_joint failed: {exc}"}
        applied_gripper: float | None = None
        if gripper is not None:
            try:
                g = max(0.0, min(1.0, float(gripper)))
            except (TypeError, ValueError):
                return {"error": "gripper must be numeric in [0, 1]"}
            applied_gripper = g
            self._sim.set_gripper("closed" if g > 0.5 else "open")
        return {
            "target_joints": list(joints),
            "applied_gripper": applied_gripper,
            "drive_real_arm": False,
        }
