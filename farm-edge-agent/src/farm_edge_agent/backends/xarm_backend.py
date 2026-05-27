"""XArmBackend — adapt the UF850 ``XArmDriver`` to ``RobotBackend``.

Differences from the sim that this layer absorbs:

* xArm SDK speaks mm + degrees; ``RobotBackend`` snapshots are mm + radians.
* xArm ``move_to`` takes a *delta*, not an absolute pose. ``jog`` builds
  the delta tuple from ``(axis, sign, step_mm, step_rad)`` and calls
  through.
* Cameras (RealSense D435 grabber) are surfaced as JPEG bytes via
  ``camera_jpeg``; no sim/placeholder fallback — when no grabber, the
  HTTP endpoint returns 503 and the dashboard tile stays black.
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
from collections import deque
from typing import Any

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
# Same idea for the ghost-arm Quest stream — looser threshold because
# the ghost moves continuously instead of in discrete clicks. Anything
# past 60° between frames is almost certainly an IK branch jump (elbow
# flip, wrist roll-over) and we drop it. Comparison is against the last
# accepted target, not the live arm joints, so the user's smooth hand
# motion doesn't keep tripping the filter while the real arm catches up.
_GHOST_MAX_JOINT_JUMP_RAD = math.radians(60.0)
# Safety valve on the branch-jump filter: if we reject this many ghost
# updates in a row, the IK is genuinely stuck in a flipped branch and
# the operator can't unstick it from the headset. Force-accept the
# next one rather than freeze indefinitely. ~0.2 s at 100 Hz Quest input.
_GHOST_REJECT_STREAK_MAX = 20

# Effectively infinite — "envelope off" mode the driver can still chew on.
_NO_ENVELOPE_MIN = (-1e9, -1e9, -1e9)
_NO_ENVELOPE_MAX = (1e9, 1e9, 1e9)

# Streaming loop — what mode 1 actually wants.
#
# xArm mode 1 (`set_servo_angle_j`) does NO interpolation in the controller:
# whatever angle we push lands as the next setpoint. If we forward each
# Quest pose directly (60–90 Hz, lumpy from Unity), the arm sees command
# gaps and stutters.
#
# Naive fix #1 — "chase the latest target with a slew cap" — replaces the
# stutter with the WORSE artifact the operator actually feels: the real
# arm falls behind, sprints at the cap velocity (faster than the hand!),
# overshoots in *velocity*, stops, sprints again. That's intrinsic to any
# catch-up strategy.
#
# What we do instead: a **fixed-delay replay buffer**. Every Quest target
# is timestamped and pushed into a deque. The 250 Hz stream loop renders
# the target sampled `_STREAM_DELAY_S` in the past, linearly interpolated
# between buffer entries. By construction the output velocity equals the
# input velocity — so the real arm tracks the digital arm's speed exactly,
# just `_STREAM_DELAY_S` seconds behind. When the operator stops, the arm
# rolls to the last buffered position and stops cleanly. No catch-up.
_STREAM_HZ = 250.0
_STREAM_PERIOD = 1.0 / _STREAM_HZ
# Tracking delay. Has to comfortably exceed the worst inter-frame gap of
# the Quest input so we always have a "future" sample to interpolate
# toward; 60 ms = ~5 frames at 90 Hz of headroom.
_STREAM_DELAY_S = 0.060
# Hard safety cap on per-cycle joint motion (~180 °/s). Sized well above
# anything the operator's hand can realistically command, so it never
# binds during normal teleop; it only catches IK glitches that escape
# the bridge's branch-jump filter.
_STREAM_SAFETY_STEP_RAD = math.radians(180.0) / _STREAM_HZ
# Cap on history retention. Anything older than render_t − this window
# is evicted each cycle. 0.5 s easily covers _STREAM_DELAY_S plus any
# clock drift.
_STREAM_HISTORY_WINDOW_S = 0.5
# One-pole IIR coefficient applied to the spline-rendered command. At
# 250 Hz this gives a ~17 Hz corner — kills residual hand tremor and
# IK micro-jitter that the spline faithfully reproduces through, while
# adding only ~6 ms of group delay on top of _STREAM_DELAY_S.
_STREAM_FILTER_ALPHA = 0.35

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
        # Drive-mode toggle. ``_drive_real_arm`` is the backing field;
        # the public ``drive_real_arm`` property has a setter that
        # transitions the xArm SDK between position-mode (mode 0, for
        # home/jog) and online-streaming mode (mode 6, for smooth Quest
        # teleop). Default off so booting can't surprise-move the arm.
        self._drive_real_arm = False
        self._streaming_active = False
        # The *digital* desired joint state — what we last commanded.
        # Diverges from ``_last_snapshot["joints"]`` while the real arm is
        # still slewing to a fresh target. The dashboard's ghost arm reads
        # this; the solid arm reads the live snapshot. Populated on the
        # first successful snapshot refresh and on every jog/home.
        self._target_joints: list[float] | None = None
        # Count of consecutive ghost-update rejections. Reset on accept;
        # forces an accept once it crosses _GHOST_REJECT_STREAK_MAX so a
        # persistent IK branch flip can't permanently freeze teleop.
        self._ghost_reject_streak: int = 0
        # 250 Hz streaming thread state. ``_stream_history`` is the
        # timestamped buffer of Quest-driven joint targets; the thread
        # samples it at (now - DELAY) and linearly interpolates to render
        # the next set_servo_angle_j. ``_stream_cmd`` is the last value
        # actually pushed (used as the safety cap's reference and for
        # re-seeding on mode transitions). Tiny separate lock so a Quest
        # pose update can append a new target without contesting the SDK
        # lock the thread is holding mid-command.
        self._stream_target_lock = threading.Lock()
        self._stream_history: deque[tuple[float, list[float]]] = deque(maxlen=512)
        self._stream_cmd: list[float] | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        # Set while jog/home/gripper temporarily drops to mode 0. The
        # stream thread checks this each tick and skips firing so we
        # never command set_servo_angle_j into the wrong mode.
        self._stream_paused = threading.Event()
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
        self._stop_stream_thread()
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
    def drive_real_arm(self) -> bool:
        return self._drive_real_arm

    @drive_real_arm.setter
    def drive_real_arm(self, value: bool) -> None:
        new_val = bool(value)
        if new_val == self._drive_real_arm:
            return
        self._drive_real_arm = new_val
        self._set_streaming_mode(new_val)

    def _set_streaming_mode(self, on: bool) -> None:
        """Switch the xArm controller between position mode (0) and servo
        joint mode (1) and start/stop the 250 Hz streaming thread.

        Mode 1 doesn't smooth its inputs — whatever angle we hand
        ``set_servo_angle_j`` becomes the next setpoint with no
        interpolation. The streaming thread owns the smoothing: it runs
        at a fixed cadence and slews toward whatever target the Quest
        bridge last published.
        """
        api = getattr(self._driver, "_api", None)
        if api is None:
            return
        if on:
            with self._lock:
                try:
                    api.motion_enable(enable=True)
                    api.set_mode(1)
                    api.set_state(0)
                    self._streaming_active = True
                    log.info("xarm: entered servo-joint mode (1)")
                except Exception as exc:
                    log.warning("xarm mode switch failed: %s", exc)
                    return
            # Start the thread *after* mode 1 is live so the first
            # set_servo_angle_j isn't fired into mode 0.
            self._start_stream_thread()
        else:
            # Stop the thread first so it doesn't try to push mode-1
            # commands after we drop back to mode 0.
            self._stop_stream_thread()
            with self._lock:
                try:
                    api.set_mode(0)
                    api.set_state(0)
                    self._streaming_active = False
                    log.info("xarm: returned to position mode (0)")
                except Exception as exc:
                    log.warning("xarm mode switch failed: %s", exc)

    def _start_stream_thread(self) -> None:
        # Seed cmd/history from live joints so toggling drive_real_arm on
        # doesn't snap the arm. The history starts with one point at
        # "now"; until a real Quest target lands, _sample_history holds
        # at this seed and the arm stays still.
        try:
            with self._lock:
                cur_deg = self._driver.read_joint_state()
            cur_rad = [math.radians(j) for j in cur_deg[:6]]
        except Exception:
            cur_rad = list(self._target_joints or [0.0] * 6)
        with self._stream_target_lock:
            self._stream_history.clear()
            self._stream_history.append((time.perf_counter(), list(cur_rad)))
            self._stream_cmd = list(cur_rad)
        self._stream_stop.clear()
        self._stream_paused.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="xarm-stream"
        )
        self._stream_thread.start()

    def _stop_stream_thread(self) -> None:
        self._stream_stop.set()
        t = self._stream_thread
        self._stream_thread = None
        if t is not None:
            t.join(timeout=0.5)

    def _stream_loop(self) -> None:
        """Fixed-rate 250 Hz loop: render the buffered target sampled at
        (now − _STREAM_DELAY_S), linearly interpolated, and push it.

        This is the *only* place set_servo_angle_j gets called. Quest pose
        callbacks just append (timestamp, joints) to ``_stream_history``;
        the loop reads from the past so output velocity = input velocity
        and the operator never feels the slew cap pulling the arm.
        """
        api = getattr(self._driver, "_api", None)
        if api is None:
            return
        next_tick = time.perf_counter()
        while not self._stream_stop.is_set():
            if not self._stream_paused.is_set():
                render_t = time.perf_counter() - _STREAM_DELAY_S
                with self._stream_target_lock:
                    target = self._sample_history_locked(render_t)
                    cmd = (
                        list(self._stream_cmd)
                        if self._stream_cmd is not None
                        else None
                    )
                if (
                    target is not None
                    and cmd is not None
                    and len(target) == 6
                    and len(cmd) == 6
                ):
                    new_cmd = [0.0] * 6
                    for i in range(6):
                        # IIR pass: filtered = cmd + α·(rendered - cmd).
                        # Removes residual high-freq noise the spline
                        # carried straight through from the buffer.
                        filtered = cmd[i] + _STREAM_FILTER_ALPHA * (
                            target[i] - cmd[i]
                        )
                        diff = filtered - cmd[i]
                        # Hard safety cap — only ever bites on an IK
                        # glitch. Normal hand motion is well below this.
                        if diff > _STREAM_SAFETY_STEP_RAD:
                            diff = _STREAM_SAFETY_STEP_RAD
                        elif diff < -_STREAM_SAFETY_STEP_RAD:
                            diff = -_STREAM_SAFETY_STEP_RAD
                        new_cmd[i] = cmd[i] + diff
                    try:
                        # Serialize SDK access via _lock so we don't
                        # interleave bytes with the poll loop's reads on
                        # the same socket. Re-check paused under the lock
                        # so a concurrent mode switch can't race.
                        with self._lock:
                            if (
                                not self._stream_paused.is_set()
                                and self._streaming_active
                            ):
                                api.set_servo_angle_j(
                                    angles=new_cmd, is_radian=True
                                )
                    except Exception as exc:
                        log.debug("stream set_servo_angle_j failed: %s", exc)
                    with self._stream_target_lock:
                        self._stream_cmd = new_cmd
            next_tick += _STREAM_PERIOD
            delay = next_tick - time.perf_counter()
            if delay > 0:
                # time.sleep releases the GIL, so the bridge's I/O
                # threads keep running.
                time.sleep(delay)
            else:
                # Behind schedule (poll loop probably held the SDK
                # lock). Reset rather than fire a catch-up burst that
                # would only feed back into jitter.
                next_tick = time.perf_counter()

    def _sample_history_locked(self, t: float) -> list[float] | None:
        """Render the buffered target at time ``t`` using a non-uniform
        Catmull-Rom spline. Caller must hold ``_stream_target_lock``.

        Why a spline instead of linear interp: linear is C⁰ continuous
        but velocity is piecewise-constant, so at every Quest-frame
        boundary (~80 Hz) the commanded velocity changes discretely.
        The arm reads that as an acceleration impulse and the operator
        feels it as buzz. Catmull-Rom is C¹ — velocity matches across
        every segment boundary, so the commanded motion has no
        synthetic high-frequency content.

        Tangents are finite-difference (m_i = (p_{i+1} - p_{i-1}) /
        (t_{i+1} - t_{i-1})), handling irregular Quest timestamps. At
        the start/end of the buffer we fall back to one-sided
        differences so the spline degenerates cleanly to Hermite-with-
        linear-tangents at the boundaries.

        Edge cases:
        * Empty buffer → None (caller skips).
        * t before oldest entry → hold at oldest (stream warmup).
        * t after newest entry → hold at newest (operator paused; arm
          coasts to the last buffered target and stops).
        """
        h = self._stream_history
        if not h:
            return None
        # Evict old entries we'll never sample again. Leave at least
        # four so the spline always has a full window around any
        # bracket point.
        cutoff = t - _STREAM_HISTORY_WINDOW_S
        while len(h) > 4 and h[1][0] < cutoff:
            h.popleft()
        n = len(h)
        if n == 1:
            return list(h[0][1])
        if t <= h[0][0]:
            return list(h[0][1])
        if t >= h[-1][0]:
            return list(h[-1][1])
        # Find the bracket: t falls in [h[i][0], h[i+1][0]].
        bracket = -1
        for k in range(n - 1):
            if h[k][0] <= t <= h[k + 1][0]:
                bracket = k
                break
        if bracket < 0:
            return list(h[-1][1])
        i = bracket
        t1, p1 = h[i]
        t2, p2 = h[i + 1]
        span = t2 - t1
        if span <= 0.0:
            return list(p2)
        # Neighbours for tangent estimation. At the buffer ends we
        # mirror so the one-sided tangent reduces to a forward /
        # backward difference (the spline degenerates to linear there).
        if i > 0:
            t0, p0 = h[i - 1]
        else:
            t0 = t1 - span
            p0 = p1
        if i + 2 < n:
            t3, p3 = h[i + 2]
        else:
            t3 = t2 + span
            p3 = p2
        u = (t - t1) / span
        u2 = u * u
        u3 = u2 * u
        # Hermite basis (cubic).
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        # Tangent denominators (Hermite expects tangents scaled to the
        # [0,1] parameterisation of the current segment).
        inv_t20 = span / (t2 - t0) if (t2 - t0) > 0.0 else 0.0
        inv_t31 = span / (t3 - t1) if (t3 - t1) > 0.0 else 0.0
        out = [0.0] * 6
        for k in range(6):
            m1 = (p2[k] - p0[k]) * inv_t20
            m2 = (p3[k] - p1[k]) * inv_t31
            out[k] = h00 * p1[k] + h10 * m1 + h01 * p2[k] + h11 * m2
        return out

    def _ensure_position_mode(self) -> bool:
        """Drop into position mode (0) so set_servo_angle / set_gripper /
        home work. Returns True if the caller should restore servo mode
        afterwards. Safe to call whether we're in streaming mode or not."""
        if not self._streaming_active:
            return False
        api = getattr(self._driver, "_api", None)
        if api is None:
            return False
        # Stop the stream from racing the mode switch — otherwise it
        # could fire one last set_servo_angle_j after we've already
        # dropped to mode 0.
        self._stream_paused.set()
        try:
            api.set_mode(0)
            api.set_state(0)
            log.info("xarm: temp drop to position mode (0)")
        except Exception as exc:
            log.warning("xarm: temp drop to position mode failed: %s", exc)
            self._stream_paused.clear()
            return False
        return True

    def _restore_servo_mode(self) -> None:
        api = getattr(self._driver, "_api", None)
        if api is None:
            return
        try:
            api.motion_enable(enable=True)
            api.set_mode(1)
            api.set_state(0)
            log.info("xarm: restored servo mode (1)")
            # Re-seed the stream from live joints so it doesn't try to
            # replay the pre-jog buffer the moment we unpause.
            try:
                cur_deg = self._driver.read_joint_state()
                cur_rad = [math.radians(j) for j in cur_deg[:6]]
                with self._stream_target_lock:
                    self._stream_cmd = list(cur_rad)
                    self._stream_history.clear()
                    self._stream_history.append(
                        (time.perf_counter(), list(cur_rad))
                    )
            except Exception:
                pass
        except Exception as exc:
            log.warning("xarm: restore servo mode failed: %s", exc)
        finally:
            self._stream_paused.clear()

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

    def camera_jpeg(self, camera: str) -> bytes | None:
        """Return the cam subprocess's JPEG bytes verbatim. The browser
        resizes with CSS ``object-fit``."""
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
            # jog uses set_servo_angle (position mode); if we're in
            # streaming mode for Quest teleop, drop back briefly.
            restore = self._ensure_position_mode()
            try:
                current_pose = self._driver.read_tcp_pose()
                target_pose = tuple(
                    float(current_pose[i]) + delta_pose[i] for i in range(6)
                )

                # IK first. If the solver rejected the pose (returns the
                # previous target unchanged), bail out — better to surface a
                # 409 than have the user click jog and see nothing happen.
                current_joints_deg = self._driver.read_joint_state()[:6]
                current_joints_rad = [math.radians(j) for j in current_joints_deg]
                target_joints, _ = self._ik_for_pose(target_pose)
                # Collapse wrap-arounds (UF850 wrist joints can sit past
                # ±180° cumulative; the SDK's IK returns canonical
                # [-π, π] which then looks like a 360° leap from the
                # current state). Without this we reject perfectly valid
                # small jogs as "elbow flips" — see the 53° false-positive
                # the operator reported.
                target_joints = self._unwrap_to(target_joints, current_joints_rad)
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
            finally:
                if restore:
                    self._restore_servo_mode()
        return {"pose": [
            snap["tcp_pos_mm"][0], snap["tcp_pos_mm"][1], snap["tcp_pos_mm"][2],
            snap["tcp_rpy"][0], snap["tcp_rpy"][1], snap["tcp_rpy"][2],
        ], "snapshot": snap}

    def set_ghost_target_pose(
        self,
        pose_mm_deg: tuple[float, float, float, float, float, float],
    ) -> dict[str, Any]:
        """Update the GHOST arm only — IK the given TCP pose, store the
        resulting joints as ``_target_joints``. The real arm is NOT
        commanded. Used by the Quest teleop bridge to preview controller
        orientation before any physical motion is allowed.

        Branch-jump filter: reject IK solutions where any joint would
        leap > 60° from the previous *target* in one frame. Compares
        against the last accepted target (not the real arm joints) so a
        smooth Quest stream stays smooth even when the real arm hasn't
        caught up yet. Catches elbow flips / wrist roll-overs that would
        let the rendered arm fold through itself.
        """
        # Soft rate-cap at 200 Hz — protects the xArm SDK from being
        # hammered if Unity ever runs hotter than the network round-trip.
        now = time.perf_counter()
        last = getattr(self, "_last_ghost_t", 0.0)
        if now - last < 1.0 / 200.0:
            return {"throttled": True}
        self._last_ghost_t = now

        target_joints, ik_ok = self._ik_for_pose(pose_mm_deg)
        if not ik_ok:
            # IK couldn't solve this pose (out of reach / singular).
            # Surface this so the bridge can count consecutive failures
            # and auto re-anchor if it persists.
            return {
                "ik_fail": True,
                "pose": list(pose_mm_deg),
            }
        # Filter elbow-flip / branch-jump solutions against the previous
        # accepted target — so smooth incremental motion stays smooth.
        prev = self._target_joints
        if prev is not None and len(prev) >= 6:
            # First, collapse spurious wrap-arounds (+179° → -179° is the
            # SAME physical joint position, just a different numerical
            # encoding). Without this, every wrist-wrap registers as a
            # 358° "jump" and the filter freezes the ghost arm even
            # though the operator is moving smoothly.
            target_joints = self._unwrap_to(target_joints, prev)
            max_delta = max(abs(target_joints[i] - prev[i]) for i in range(6))
            if max_delta > _GHOST_MAX_JOINT_JUMP_RAD:
                self._ghost_reject_streak += 1
                if self._ghost_reject_streak < _GHOST_REJECT_STREAK_MAX:
                    return {
                        "rejected": "ik_branch_jump",
                        "max_delta_deg": math.degrees(max_delta),
                        "streak": self._ghost_reject_streak,
                    }
                # Safety valve fired — fall through and accept this one.
                # Better a single big jump than a permanent freeze.
                log.warning(
                    "ghost force-accepting after %d rejections "
                    "(max_delta=%.1f°) — IK branch persistent",
                    self._ghost_reject_streak,
                    math.degrees(max_delta),
                )
        self._ghost_reject_streak = 0

        with self._lock:
            self._target_joints = target_joints
            self._last_snapshot["target_joints"] = list(target_joints)

        # Append to the streaming thread's replay buffer. The loop will
        # render this (interpolated) _STREAM_DELAY_S from now. See
        # _stream_loop / _sample_history_locked.
        if self._drive_real_arm and not self._estopped:
            with self._stream_target_lock:
                self._stream_history.append(
                    (time.perf_counter(), list(target_joints))
                )
        return {"target_joints": list(target_joints), "pose": list(pose_mm_deg)}

    def home(self) -> dict[str, Any]:
        self._require_armed()
        with self._lock:
            # home() uses set_servo_angle (position mode), which only
            # works in mode 0. If we're currently streaming (mode 1 for
            # Quest teleop), drop briefly to 0 and restore afterwards.
            restore = self._ensure_position_mode()
            try:
                target_joints, _ = self._ik_for_pose(DEFAULT_HOME_POSE_850)
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
                snap = self._refresh_snapshot_locked()
            finally:
                if restore:
                    self._restore_servo_mode()
            return snap

    @staticmethod
    def _unwrap_to(joints: list[float], prev: list[float]) -> list[float]:
        """Shift each joint by an integer multiple of 2π so it lands
        within ±π of ``prev``. The xArm onboard IK returns the canonical
        [-π, π] solution; if a joint is near the wrap boundary, two
        consecutive calls can flip sign and look like a 2π jump. After
        this normalisation, the delta reflects only the physical motion.
        """
        out = list(joints)
        n = min(len(out), len(prev))
        twopi = 2.0 * math.pi
        for i in range(n):
            d = out[i] - prev[i]
            if d > math.pi or d < -math.pi:
                # round-to-nearest multiple of 2π — handles multi-wrap.
                out[i] -= twopi * round(d / twopi)
        return out

    def _ik_for_pose(
        self, pose_mm_deg: tuple[float, ...]
    ) -> tuple[list[float], bool]:
        """Query the xArm's onboard IK. Returns ``(joints, ok)``: on
        failure ``joints`` falls back to the last known target so the
        caller can keep going, but ``ok`` is False so it can also tell
        that nothing moved. Bridge uses ``ok`` to count consecutive
        stalls and trigger an auto re-anchor. SDK call is one TCP
        roundtrip; serialized via ``_lock``."""
        api: Any = getattr(self._driver, "_api", None)
        if api is None:
            return list(self._target_joints or [0.0] * 6), False
        try:
            with self._lock:
                code, angles = api.get_inverse_kinematics(
                    list(pose_mm_deg),
                    input_is_radian=False,
                    return_is_radian=True,
                )
            if code != 0 or not angles:
                log.debug("IK rejected pose=%s code=%s", pose_mm_deg, code)
                return list(self._target_joints or [0.0] * 6), False
            return [float(a) for a in angles[:6]], True
        except Exception as exc:  # noqa: BLE001
            log.debug("IK call raised: %s", exc)
            return list(self._target_joints or [0.0] * 6), False

    def set_gripper(self, state: GripperState) -> dict[str, Any]:
        self._require_armed()
        with self._lock:
            # Gripper SDK calls expect position mode (0). If we're
            # streaming (Quest teleop in mode 1), bounce briefly.
            restore = self._ensure_position_mode()
            try:
                # wait=False so the poll thread can refresh gripper
                # position (now read from the SDK) while the gripper is
                # mid-travel — the rendered fingers in the dashboard
                # animate live.
                self._driver.set_gripper(state, wait=False)
                snap = self._refresh_snapshot_locked()
                snap["gripper"] = state
                self._last_snapshot["gripper"] = state
            finally:
                if restore:
                    self._restore_servo_mode()
        return snap

    # ── safety ────────────────────────────────────────────────────────────

    def estop(self) -> dict[str, Any]:
        """Software e-stop: ask the SDK to halt motion and set our gate."""
        log.warning("E-STOP triggered")
        self._estopped = True
        # Block the stream first so we don't fire one last command into
        # a halted controller.
        self._stream_paused.set()
        try:
            api = getattr(self._driver, "_api", None)
            if api is not None and hasattr(api, "set_state"):
                with self._lock:
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
                with self._lock:
                    api.set_state(_STATE_READY)
        except Exception as exc:
            log.error("set_state(ready) failed: %s", exc)
        self._estopped = False
        # Re-seed the stream from live joints so unpausing doesn't replay
        # the stale pre-estop buffer.
        if self._streaming_active:
            try:
                with self._lock:
                    cur_deg = self._driver.read_joint_state()
                cur_rad = [math.radians(j) for j in cur_deg[:6]]
                with self._stream_target_lock:
                    self._stream_cmd = list(cur_rad)
                    self._stream_history.clear()
                    self._stream_history.append(
                        (time.perf_counter(), list(cur_rad))
                    )
            except Exception:
                pass
        self._stream_paused.clear()
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
        """Background-poll path: do the SDK reads each under its own
        brief lock acquisition so the streaming thread never waits more
        than one TCP roundtrip per missed cycle. The earlier monolithic
        ``_refresh_snapshot_locked`` held _lock across all three reads
        (~6–10 ms), which translated into a visible micro-pause in the
        arm motion every poll tick."""
        try:
            with self._lock:
                joints_deg = self._driver.read_joint_state()
            with self._lock:
                pose_mm_deg = self._driver.read_tcp_pose()
            with self._lock:
                raw_grip = self._driver.read_gripper_position()
        except XArmDriverError as exc:
            log.debug("snapshot refresh failed: %s", exc)
            with self._lock:
                return dict(self._last_snapshot)
        # Pure compute (no SDK roundtrips) — outside the lock.
        joints_rad = [math.radians(j) for j in joints_deg[:6]]
        tcp_pos_mm = [
            float(pose_mm_deg[0]),
            float(pose_mm_deg[1]),
            float(pose_mm_deg[2]),
        ]
        tcp_rpy = [math.radians(float(v)) for v in pose_mm_deg[3:6]]
        if not math.isnan(raw_grip):
            grip01: float | None = max(0.0, min(1.0, 1.0 - raw_grip / 850.0))
        else:
            grip01 = None  # fall back to cached value below
        # Single brief acquisition to publish the new snapshot dict.
        with self._lock:
            if self._target_joints is None:
                self._target_joints = list(joints_rad)
            self._last_snapshot = {
                "joints": joints_rad,
                "target_joints": list(self._target_joints),
                "tcp_pos_mm": tcp_pos_mm,
                "tcp_rpy": tcp_rpy,
                "gripper": self._last_snapshot.get("gripper", "open"),
                "gripper_pos": (
                    grip01
                    if grip01 is not None
                    else self._last_snapshot.get("gripper_pos", 0.0)
                ),
                "t": time.time(),
                "backend": self.backend_name,
                "arm_ip": self._arm_ip,
                "estopped": self._estopped,
                "cameras": self.cameras,
            }
            return dict(self._last_snapshot)

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


