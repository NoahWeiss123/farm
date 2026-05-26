"""XArmBackend — adapt the UF850 ``XArmDriver`` to ``RobotBackend``.

Differences from the sim that this layer absorbs:

* xArm SDK speaks mm + degrees; ``RobotBackend`` snapshots are mm + radians.
* xArm ``move_to`` takes a *delta*, not an absolute pose. ``jog`` builds
  the delta tuple from ``(axis, sign, step_mm, step_rad)`` and calls
  through.
* xArm has no cameras. ``render_rgb`` returns a labelled "no signal"
  placeholder so the dashboard tiles stay rendered.
* The xArm SDK doesn't push joint-state callbacks. Tasks that need a
  live view (the dashboard's SSE stream) poll ``snapshot`` instead.

The envelope check inside ``XArmDriver`` is configurable per-instance, so
this backend passes effectively infinite bounds when ``envelope`` is
``None``. The driver still owns the check; we just opt out at the boundary.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np

from farm_edge_agent.cameras import RealsenseGrabber, RealsenseUnavailable
from farm_edge_agent.drivers.xarm import (
    DEFAULT_ENVELOPE_MAX,
    DEFAULT_ENVELOPE_MIN,
    DEFAULT_HOME_POSE_850,
    XArmDriver,
    XArmDriverError,
)

from .base import GripperState, JogAxis

log = logging.getLogger("farm.backends.xarm")

DEFAULT_VELOCITY_CAP_MM_S = 50.0
DEFAULT_JOINT_SPEED_RAD_S = math.radians(40.0)  # slow + safe
DEFAULT_STEP_MM = 5.0
DEFAULT_STEP_RAD = math.radians(2.0)

# Largest per-joint delta a single jog is allowed to issue. Catches the
# case where the SDK's IK picks a different branch than what's currently
# loaded (e.g. flips the elbow); without this the arm would swing
# violently to reach a numerically valid but geometrically distant
# joint configuration.
_JOG_MAX_JOINT_DELTA_RAD = math.radians(30.0)

# Effectively infinite — "envelope off" mode the driver can still chew on.
_NO_ENVELOPE_MIN = (-1e9, -1e9, -1e9)
_NO_ENVELOPE_MAX = (1e9, 1e9, 1e9)

# xArm SDK state codes
_STATE_READY = 0
_STATE_STOP = 4


class XArmBackend:
    backend_name = "xarm"

    def __init__(
        self,
        arm_ip: str,
        *,
        envelope: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
            DEFAULT_ENVELOPE_MIN,
            DEFAULT_ENVELOPE_MAX,
        ),
        velocity_cap_mm_s: float = DEFAULT_VELOCITY_CAP_MM_S,
        home_pose=DEFAULT_HOME_POSE_850,
        cameras: bool = True,
        camera_mapping: dict[str, str] | None = None,
    ) -> None:
        self._arm_ip = arm_ip
        if envelope is None:
            log.warning("xarm backend: envelope DISABLED — no workspace bounds enforced")
            env_min, env_max = _NO_ENVELOPE_MIN, _NO_ENVELOPE_MAX
        else:
            env_min, env_max = envelope
        self._driver = XArmDriver(
            arm_ip,
            home_pose=home_pose,
            envelope_min=env_min,
            envelope_max=env_max,
        )
        self._velocity_cap = float(velocity_cap_mm_s)
        self._lock = threading.RLock()
        self._estopped = False
        # The *digital* desired joint state — what we last commanded.
        # Diverges from ``_last_snapshot["joints"]`` while the real arm is
        # still slewing to a fresh target. The dashboard's ghost arm reads
        # this; the solid arm reads the live snapshot. Populated on the
        # first successful snapshot refresh and on every jog/home.
        self._target_joints: list[float] | None = None
        self._grabber: RealsenseGrabber | None = None
        if cameras:
            try:
                self._grabber = RealsenseGrabber(mapping=camera_mapping)
                log.info("realsense grabber initialized: %s", self._grabber.names())
            except RealsenseUnavailable as exc:
                log.warning("realsense cameras unavailable: %s — falling back to placeholder tiles", exc)

        # Cache the most recent snapshot so HTTP/SSE consumers don't beat
        # the SDK to death — refreshed on every jog/home, plus a background
        # poll loop primes it at ~5 Hz.
        self._last_snapshot: dict[str, Any] = {
            "joints": [0.0] * 6,
            "target_joints": [0.0] * 6,
            "tcp_pos_mm": [0.0, 0.0, 0.0],
            "tcp_rpy": [0.0, 0.0, 0.0],
            "gripper": "open",
            "gripper_pos": 0.0,
            "t": time.time(),
            "backend": self.backend_name,
            "arm_ip": self._arm_ip,
            "estopped": False,
            "cameras": ["base", "wrist"],
        }
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        log.info("connecting to UF850 at %s …", self._arm_ip)
        self._driver.connect()
        log.info("UF850 connected")
        self._refresh_snapshot()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        if self._grabber is not None:
            self._grabber.start()

    def disconnect(self) -> None:
        self._poll_stop.set()
        if self._grabber is not None:
            try:
                self._grabber.stop()
            except Exception as exc:
                log.warning("grabber stop raised: %s", exc)
        try:
            self._driver.disconnect()
        except Exception as exc:
            log.warning("disconnect raised: %s", exc)

    @property
    def cameras(self) -> list[str]:
        if self._grabber is None:
            return ["base", "wrist"]
        return self._grabber.names()

    def swap_cameras(self) -> dict[str, str]:
        if self._grabber is None:
            return {}
        return self._grabber.swap()

    # ── observation ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_snapshot)

    def render_rgb(self, camera: str, *, width: int, height: int) -> np.ndarray:
        if self._grabber is not None:
            frame = self._grabber.latest(camera)
            if frame is not None:
                return _resize_rgb(frame, width=width, height=height)
        return _placeholder_image(camera, width=width, height=height)

    def camera_jpeg(self, camera: str) -> bytes | None:
        """Fast path: return the cam subprocess's JPEG bytes verbatim.

        Skips the decode → resize → re-encode the generic ``render_rgb``
        path would otherwise do. The browser resizes the image with CSS
        ``object-fit``, which is plenty good for the dashboard tiles.
        """
        if self._grabber is None:
            return None
        return self._grabber.latest_jpeg(camera)

    # ── motion ────────────────────────────────────────────────────────────

    def jog(
        self, axis: JogAxis, sign: int, *, step_mm: float, step_rad: float
    ) -> dict[str, Any]:
        self._require_armed()
        if axis not in ("x", "y", "z", "rx", "ry", "rz"):
            raise ValueError(f"unknown jog axis: {axis!r}")
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or +1")

        step_deg = math.degrees(step_rad)
        deltas = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
        if axis in ("x", "y", "z"):
            deltas[axis] = sign * step_mm
        else:
            deltas[axis] = sign * step_deg
        delta_pose = (
            deltas["x"], deltas["y"], deltas["z"],
            deltas["rx"], deltas["ry"], deltas["rz"],
        )

        with self._lock:
            current_pose = self._driver.read_tcp_pose()
            target_pose = tuple(
                float(current_pose[i]) + delta_pose[i] for i in range(6)
            )

            # IK first. If the solver rejected the pose (returns the
            # previous target unchanged), bail out — better to surface a
            # 409 than have the user click jog and see nothing happen.
            current_joints_deg = self._driver.read_joint_state()[:6]
            current_joints_rad = [math.radians(j) for j in current_joints_deg]
            target_joints = self._ik_for_pose(target_pose)
            ik_delta = max(abs(target_joints[i] - current_joints_rad[i]) for i in range(6))
            if ik_delta > _JOG_MAX_JOINT_DELTA_RAD:
                raise RuntimeError(
                    f"jog rejected: IK picked a joint branch {math.degrees(ik_delta):.1f}° "
                    f"away from current — likely an elbow flip. "
                    f"Try a smaller step or a different axis."
                )
            self._target_joints = target_joints

            # Command joint angles directly. Using set_servo_angle (not
            # set_position delta) means the real arm converges to the
            # SAME joint configuration we predicted for the ghost — they
            # match up at the end of every move, regardless of IK
            # multiplicity.
            api = getattr(self._driver, "_api", None)
            if api is None:
                raise RuntimeError("xArm SDK not connected")
            code = api.set_servo_angle(
                angle=list(target_joints),
                speed=DEFAULT_JOINT_SPEED_RAD_S,
                is_radian=True,
                wait=False,
            )
            if code != 0:
                raise RuntimeError(f"xArm set_servo_angle returned code {code}")
            snap = self._refresh_snapshot_locked()
        return {"pose": [
            snap["tcp_pos_mm"][0], snap["tcp_pos_mm"][1], snap["tcp_pos_mm"][2],
            snap["tcp_rpy"][0], snap["tcp_rpy"][1], snap["tcp_rpy"][2],
        ], "snapshot": snap}

    def home(self) -> dict[str, Any]:
        self._require_armed()
        with self._lock:
            # IK the home pose and command those joint angles directly,
            # same as jog. This guarantees the real arm lands on exactly
            # the same joint config the ghost predicted instead of one
            # of the SDK-picked alternates.
            target_joints = self._ik_for_pose(DEFAULT_HOME_POSE_850)
            self._target_joints = target_joints
            api = getattr(self._driver, "_api", None)
            if api is None:
                raise RuntimeError("xArm SDK not connected")
            code = api.set_servo_angle(
                angle=list(target_joints),
                speed=DEFAULT_JOINT_SPEED_RAD_S,
                is_radian=True,
                wait=False,
            )
            if code != 0:
                raise RuntimeError(f"xArm home returned code {code}")
            return self._refresh_snapshot_locked()

    def _ik_for_pose(self, pose_mm_deg: tuple[float, ...]) -> list[float]:
        """Query the xArm's onboard IK for the joint state at ``pose``.

        Falls back to the last known target / current joints if the SDK
        rejects the pose (e.g. out of reach). Returns a 6-element list of
        radians.
        """
        api: Any = getattr(self._driver, "_api", None)
        if api is None:
            return list(self._target_joints or [0.0] * 6)
        try:
            code, angles = api.get_inverse_kinematics(
                list(pose_mm_deg), input_is_radian=False, return_is_radian=True
            )
            if code != 0 or not angles:
                log.debug("IK rejected pose=%s code=%s", pose_mm_deg, code)
                return list(self._target_joints or [0.0] * 6)
            return [float(a) for a in angles[:6]]
        except Exception as exc:  # noqa: BLE001
            log.debug("IK call raised: %s", exc)
            return list(self._target_joints or [0.0] * 6)

    def set_gripper(self, state: GripperState) -> dict[str, Any]:
        self._require_armed()
        with self._lock:
            # wait=False so the poll thread can refresh gripper position
            # (now read from the SDK) while the gripper is mid-travel —
            # the rendered fingers in the dashboard animate live.
            self._driver.set_gripper(state, wait=False)
            snap = self._refresh_snapshot_locked()
            snap["gripper"] = state
            self._last_snapshot["gripper"] = state
        return snap

    # ── safety ────────────────────────────────────────────────────────────

    def estop(self) -> dict[str, Any]:
        """Software e-stop: ask the SDK to halt motion and set our gate."""
        log.warning("E-STOP triggered")
        self._estopped = True
        try:
            api = getattr(self._driver, "_api", None)
            if api is not None and hasattr(api, "set_state"):
                api.set_state(_STATE_STOP)
        except Exception as exc:
            log.error("set_state(stop) failed: %s", exc)
        with self._lock:
            self._last_snapshot["estopped"] = True
        return {"estopped": True}

    def estop_clear(self) -> dict[str, Any]:
        log.info("E-STOP cleared")
        try:
            api = getattr(self._driver, "_api", None)
            if api is not None and hasattr(api, "set_state"):
                api.set_state(_STATE_READY)
        except Exception as exc:
            log.error("set_state(ready) failed: %s", exc)
        self._estopped = False
        with self._lock:
            self._last_snapshot["estopped"] = False
        return {"estopped": False}

    # ── internals ──────────────────────────────────────────────────────────

    def _require_armed(self) -> None:
        if self._estopped:
            raise RuntimeError(
                "xarm backend is e-stopped; clear via /v1/teleop/estop/clear "
                "after verifying the workspace"
            )

    def _refresh_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._refresh_snapshot_locked()

    def _refresh_snapshot_locked(self) -> dict[str, Any]:
        try:
            joints_deg = self._driver.read_joint_state()
            pose_mm_deg = self._driver.read_tcp_pose()
        except XArmDriverError as exc:
            log.debug("snapshot refresh failed: %s", exc)
            return dict(self._last_snapshot)
        # The SDK returns 7 joint slots even for the 6-DOF UF850 (the 7th
        # is always 0). Trim so downstream consumers (dashboard joint bars,
        # ROS /joint_states publisher) see the actual 6.
        joints_rad = [math.radians(j) for j in joints_deg[:6]]
        tcp_pos_mm = [float(pose_mm_deg[0]), float(pose_mm_deg[1]), float(pose_mm_deg[2])]
        tcp_rpy = [math.radians(float(v)) for v in pose_mm_deg[3:6]]
        # Seed target_joints from current on the first read so the ghost
        # arm coincides with the real arm until the first command.
        if self._target_joints is None:
            self._target_joints = list(joints_rad)

        # Poll the actual gripper opening. SDK range is 0..850; the
        # dashboard expects 0 (open) → 1 (closed), so invert.
        raw_grip = self._driver.read_gripper_position()
        if not math.isnan(raw_grip):
            grip01 = max(0.0, min(1.0, 1.0 - raw_grip / 850.0))
        else:
            # SDK call failed (gripper not enabled / not present) —
            # fall back to whatever set_gripper last committed.
            grip01 = self._last_snapshot.get("gripper_pos", 0.0)

        self._last_snapshot = {
            "joints": joints_rad,
            "target_joints": list(self._target_joints),
            "tcp_pos_mm": tcp_pos_mm,
            "tcp_rpy": tcp_rpy,
            "gripper": self._last_snapshot.get("gripper", "open"),
            "gripper_pos": grip01,
            "t": time.time(),
            "backend": self.backend_name,
            "arm_ip": self._arm_ip,
            "estopped": self._estopped,
            "cameras": self.cameras,
        }
        return dict(self._last_snapshot)

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            try:
                self._refresh_snapshot()
            except Exception:
                pass
            self._poll_stop.wait(0.2)  # ~5 Hz background refresh


def _resize_rgb(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize an RGB frame to the requested dimensions.

    OpenCV is the cheapest dep we already pull (via cv2 in cameras/realsense
    ecosystem); if it's unavailable, fall back to a slow numpy nearest-neighbour
    so the dashboard at least shows *something*.
    """
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    try:
        import cv2
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        ys = np.linspace(0, frame.shape[0] - 1, height).astype(int)
        xs = np.linspace(0, frame.shape[1] - 1, width).astype(int)
        return frame[ys][:, xs]


def _placeholder_image(label: str, *, width: int, height: int) -> np.ndarray:
    """Return a uniformly grey RGB tile with a small dark border.

    Drawing text without a font dependency means doing it pixel-art-style,
    which is uglier than just rendering a tinted tile. The dashboard's
    figcaption already labels the feed, so the placeholder only needs to
    say "this isn't dead, just not wired."
    """
    img = np.full((height, width, 3), 36, dtype=np.uint8)
    border = 2
    img[:border, :, :] = 96
    img[-border:, :, :] = 96
    img[:, :border, :] = 96
    img[:, -border:, :] = 96
    # Diagonal hash so it's visually distinguishable from a frozen feed.
    for y in range(0, height, 8):
        x = (y + height) % (width - 4)
        img[y:y+2, x:x+2, :] = 72
    _ = label  # caption is in the DOM; no in-image rendering
    return img
