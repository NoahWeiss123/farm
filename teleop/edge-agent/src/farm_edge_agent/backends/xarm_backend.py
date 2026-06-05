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
    DEFAULT_HOME_JOINTS_850_DEG,
    DEFAULT_HOME_POSE_850,
    XArmDriver,
    XArmDriverError,
)
from farm_edge_agent.one_euro import OneEuroFilterND

from .base import GripperState, JogAxis

log = logging.getLogger("farm.backends.xarm")

DEFAULT_VELOCITY_CAP_MM_S = 50.0
DEFAULT_JOINT_SPEED_RAD_S = math.radians(120.0)  # 3× of the prior 40°/s
DEFAULT_STEP_MM = 5.0
DEFAULT_STEP_RAD = math.radians(2.0)
# Joint acceleration cap for discrete moves (jog/home). The xArm SDK's
# default mvacc is ~17 rad/s², which produces a near-step velocity
# change at the end of every move — the operator sees the arm reach
# target and visibly snap to a halt. Setting mvacc to ~1 rad/s² (≈57°/s²)
# gives a ~0.7 s deceleration ramp on top of DEFAULT_JOINT_SPEED_RAD_S,
# so the arm coasts into the target instead of locking.
DEFAULT_JOINT_ACCEL_RAD_S2 = math.radians(60.0)

# Homing is a discrete, operator-initiated one-shot move — there's no
# fine-control feel to preserve, just "get there fast". Decoupled from
# the jog/teleop DEFAULT_JOINT_* tune (which is deliberately slow for
# soft deceleration on small inputs). 10× the previous tune (was
# 200°/s, 800°/s²) per operator request — the xArm controller will
# clip to its per-joint hardware limits anyway, so commanding higher
# just guarantees we saturate those.
HOMING_JOINT_SPEED_RAD_S = math.radians(2000.0)  # ~35 rad/s commanded
HOMING_JOINT_ACCEL_RAD_S2 = math.radians(8000.0)  # ~140 rad/s² commanded

# Effectively infinite — "envelope off" mode the driver can still chew on.
_NO_ENVELOPE_MIN = (-1e9, -1e9, -1e9)
_NO_ENVELOPE_MAX = (1e9, 1e9, 1e9)

# Streaming loop: replay the ghost arm's recorded trajectory.
#
# Every ghost-target update appends (t, joints) to a deque. The 250 Hz
# stream loop renders that history at (now − _STREAM_DELAY_S), linearly
# interpolated between adjacent samples. No setpoint chasing, no
# smoothing — the real arm walks through the same joint positions the
# ghost did, at the ghost's actual frame-to-frame velocity, just
# _STREAM_DELAY_S seconds behind. When the ghost stops, the renderer
# reaches the last entry and holds; the real arm stops too.
_STREAM_HZ = 250.0
_STREAM_PERIOD = 1.0 / _STREAM_HZ
# Tracking delay. Has to comfortably exceed the worst inter-frame gap of
# the Quest input so the renderer always has a "future" sample to
# interpolate toward. 30 ms = ~3 frames at 90 Hz of headroom; tighter
# than this and Quest jitter can have the renderer briefly walk past
# the newest sample, causing micro-stutter. (Was 60 ms — halved to cut
# perceived button-to-action latency.)
_STREAM_DELAY_S = 0.030
# Software velocity cap on the streaming path — set effectively
# unbounded (50000°/s ≈ 873 rad/s, ~6× the worst-case Kp·err the PD
# tracker can request, so it never bites). Per user request: when the
# operator moves fast, the software cap was producing a stop-start
# rhythm because the tracker couldn't follow and stalled out at the
# limit. The xArm controller's per-joint hardware limits remain the
# real safety floor.
_STREAM_SAFETY_STEP_RAD = math.radians(50000.0) / _STREAM_HZ
# Cap on history retention. Anything older than render_t − this window
# is evicted each cycle. 0.5 s easily covers _STREAM_DELAY_S plus any
# clock drift.
_STREAM_HISTORY_WINDOW_S = 0.5

# ── Policy actuation interpolation ───────────────────────────────────────
# A learned policy POSTs ABSOLUTE joint targets at the training timestep
# (~30 Hz). The 250 Hz stream loop runs ~8× faster, so emitting each target
# verbatim re-commands the SAME setpoint ~8 ticks in a row, then snaps to the
# next — a 30 Hz staircase the operator sees as jumping. Instead we linearly
# interpolate from the last commanded position toward the newest target across
# the measured inter-target interval, emitting a fresh setpoint every 250 Hz
# tick. The arm receives a smooth ≥60 Hz command stream yet still passes
# through every waypoint the policy commanded — NO 1€ filter and NO PD chase
# (those shaped the achieved-state training labels, so re-applying them
# double-lags the arm; that was the bug we removed).
_POLICY_TARGET_HZ = 30.0                 # nominal policy POST rate
_POLICY_INTERP_NOMINAL_S = 1.0 / _POLICY_TARGET_HZ
# Only fold an observed inter-arrival gap into the horizon EMA if it looks
# like a steady cadence (not a chunk-boundary inference stall). Stalls would
# inflate the horizon and leave the arm persistently lagging its waypoints.
_POLICY_INTERP_MIN_S = 1.0 / 120.0       # ignore sub-8 ms bursts
_POLICY_INTERP_EMA_MAX_S = 0.060         # gaps above this = stall, excluded
_POLICY_INTERP_EMA_ALPHA = 0.3           # EMA weight on each fresh interval
# Hard clamp on the horizon used by the lerp, so a pathological measurement
# can never freeze the arm or make it lunge.
_POLICY_INTERP_CLAMP_MIN_S = 1.0 / 120.0
_POLICY_INTERP_CLAMP_MAX_S = 0.080

# When render_t passes the newest buffered sample (Quest input briefly
# stalled — network hiccup, frame drop, controller momentarily idle),
# carry the last segment's velocity forward for up to this many seconds
# instead of hard-snapping to the held sample value. The extrapolation
# is capped at one full segment span, so the worst-case overshoot is
# bounded even if the operator truly stopped between the last two
# samples. Removes the "reach a sample, stop, jump to next" feel.
_STREAM_EXTRAP_S = 0.060

# Acceleration limiter (critically-damped second-order tracker).
#
# Without this the stream loop commands "go to rendered target NOW";
# every Quest-frame boundary is a velocity step, and the operator feels
# that as buzz. Just clamping Δvel/cycle (a bare accel cap) leaves an
# undamped second-order system that wobbles after trigger release.
# Solution: PD tracker chosen for critical damping, plus a hard accel
# clamp on the controller's output.
#
#   desired_accel = Kp·(target − cmd) − Kd·current_vel
#   clamped_accel = clamp(desired_accel, ±_STREAM_MAX_ACCEL_RAD_S2)
#   new_vel       = current_vel + clamped_accel · dt
#   new_cmd       = cmd + new_vel · dt
#
# Kp = ω², Kd = 2ω. ω = 15 rad/s → 4/ω ≈ 270 ms step-input settling.
# Tuned down from 60 rad/s so the arm coasts into the final target on
# trigger release instead of snapping to a halt — the operator sees a
# visible ease-in over ~¼ second. Steady-state ramp-tracking lag =
# Kd·V/Kp = (2/ω)·V ≈ 0.133·V rad ≈ 7.6° per (rad/s) of joint speed;
# perceptible but acceptable for the smoother stop.
#
# This is the QUEST-tuned default. The policy-driving path swaps to a
# faster ω (see _STREAM_OMEGA_POLICY_RAD_S below) so the arm tracks the
# policy's intended joint trajectory at full speed instead of being
# lowpass-attenuated by the slow Quest tracker.
_STREAM_OMEGA_RAD_S = 15.0
_STREAM_KP = _STREAM_OMEGA_RAD_S * _STREAM_OMEGA_RAD_S
_STREAM_KD = 2.0 * _STREAM_OMEGA_RAD_S
# Policy-driven tracker. ω = 30 rad/s → 4/ω ≈ 133 ms settling, ~3 Hz
# tracking bandwidth. ~2× the Quest-tuned bandwidth so the arm actually
# moves at the policy's commanded velocity instead of getting
# lowpass-smoothed below recognisable. Critically damped (Kd = 2ω) so
# it never overshoots grasp targets. Steady-state ramp lag = (2/ω)·V
# ≈ 0.067·V rad ≈ 3.8° per (rad/s). The previous attempt at ω=35
# oscillated on hardware (likely exciting arm/payload resonance); 30
# keeps a safe margin from that.
_STREAM_OMEGA_POLICY_RAD_S = 30.0
_STREAM_KP_POLICY = _STREAM_OMEGA_POLICY_RAD_S * _STREAM_OMEGA_POLICY_RAD_S
_STREAM_KD_POLICY = 2.0 * _STREAM_OMEGA_POLICY_RAD_S
# Software acceleration cap on the PD tracker — effectively unbounded
# (5000 rad/s², ~4× the worst-case Kp·err of the PD tracker). The
# previous low cap (15 rad/s²) made fast hand motion feel like it was
# "stopping" because the tracker couldn't ramp velocity fast enough
# to keep up. Soft-deceleration character on small inputs is still
# preserved by the PD tracker's natural critical damping at ω=15 —
# Kp·err stays small near steady state so the clamp never enters
# the picture there.
_STREAM_MAX_ACCEL_RAD_S2 = 5000.0

# 1€ filter (Casiez/Roussel/Vogel CHI 2012) applied to the IK'd joint
# vector before it enters the stream-history buffer. Quest pose
# updates land at ~90 Hz with sub-degree jitter even when the user's
# hand is still; that jitter gets amplified by IK (small TCP jiggle →
# larger joint jiggle on near-singular configurations) and then
# faithfully replayed onto the arm by the streaming loop. The filter
# strips the high-frequency content at rest while letting fast hand
# motion through with minimal lag.
#
#   min_cutoff = 1.5 Hz → moderate smoothing at rest. Higher than the
#                         canonical 1.0 to keep the arm feeling
#                         responsive on small inputs; the Catmull-Rom
#                         interpolation downstream picks up most of
#                         the residual smoothness without adding lag.
#   beta       = 0.10   → cutoff rises ~5.7 Hz per (rad/s) of joint
#                         velocity, so fast motion gets essentially no
#                         smoothing (cutoff far above signal band).
#                         Upper end of the canonical 0–0.1 range.
#   d_cutoff   = 1.0 Hz → canonical default for the derivative.
#
# Tune these if the arm feels too laggy (raise both) or too jittery
# (lower min_cutoff). Values verified by ear/eye on the UF850; the
# paper's defaults for mouse cursors are a reasonable starting point.
_JOINT_FILTER_MIN_CUTOFF_HZ = 1.5
_JOINT_FILTER_BETA = 0.10
_JOINT_FILTER_D_CUTOFF_HZ = 1.0
# Max commanded joint velocity (rad/s) — same envelope as the existing
# safety step cap, just expressed in velocity units for the tracker.
_STREAM_MAX_VEL_RAD_S = _STREAM_SAFETY_STEP_RAD / _STREAM_PERIOD

# xArm SDK state codes
_STATE_READY = 0
_STATE_STOP = 4


def _catmull_rom6(
    p0: list[float],
    p1: list[float],
    p2: list[float],
    p3: list[float],
    alpha: float,
) -> list[float]:
    """Uniform Catmull-Rom cubic at parameter ``alpha`` ∈ [0,1] on the
    segment between ``p1`` (alpha=0) and ``p2`` (alpha=1).

    Tangents at the segment endpoints are derived from the neighbor
    samples ``p0`` and ``p3``: m₁ = (p₂ − p₀)/2, m₂ = (p₃ − p₁)/2. The
    result is C¹-continuous across segments — joining segments share
    both position and velocity — so the rendered trajectory has no
    velocity steps at Quest-frame boundaries the way bare linear
    interpolation does.

    Specialized to 6 channels (the UF850 joint vector) to keep the
    inner loop allocation-free in the 250 Hz stream thread.
    """
    a2 = alpha * alpha
    a3 = a2 * alpha
    out = [0.0] * 6
    for k in range(6):
        out[k] = 0.5 * (
            (2.0 * p1[k])
            + (-p0[k] + p2[k]) * alpha
            + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * a2
            + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * a3
        )
    return out


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
        # Which upstream pushed the most recent stream sample. The
        # stream loop reads this each tick and picks Quest-tuned (ω=15)
        # vs. policy-tuned (ω=25) PD gains. Keeps Quest's "ease-in"
        # feel while letting policy commands run at full speed.
        self._policy_drive_active = False
        # Latest RAW absolute joint target from a policy (set by
        # set_joint_target). When _policy_drive_active is True the 250 Hz stream
        # loop LINEARLY INTERPOLATES from _policy_start toward this target across
        # _policy_interval_s — NO 1€ filter, NO Catmull-Rom, NO PD chase — so the
        # arm smoothly but faithfully reproduces the model's commanded
        # trajectory. (Those filters already shaped the achieved-state training
        # labels; re-applying them at inference double-lags the arm.)
        self._policy_target: list[float] | None = None
        # Interpolation state for the smooth policy path (see _stream_loop):
        #   _policy_start      — commanded position when _policy_target landed
        #                        (= the live _stream_cmd, so motion is C0 across
        #                        waypoints)
        #   _policy_target_t   — perf_counter when _policy_target landed
        #   _policy_interval_s — EMA of the inter-arrival gap = the lerp horizon
        #   _policy_last_arrival — perf_counter of the previous target (for the
        #                          inter-arrival measurement)
        self._policy_start: list[float] | None = None
        self._policy_target_t: float | None = None
        self._policy_interval_s: float = _POLICY_INTERP_NOMINAL_S
        self._policy_last_arrival: float | None = None
        # Most recent gripper SDK position commanded (0..850 raw). The
        # policy path sends a gripper value per chunk action (≥30 Hz)
        # but the parallel gripper hardware takes ~500 ms to travel a
        # full range — spamming the SDK at 30 Hz with near-identical
        # values keeps re-targeting the controller and slows the close
        # cycle. Skip the SDK call when the new value differs by < 5 %
        # of full travel from the last one sent.
        self._last_gripper_sdk_pos: int | None = None
        # The *digital* desired joint state — what we last commanded.
        # Diverges from ``_last_snapshot["joints"]`` while the real arm is
        # still slewing to a fresh target. The dashboard's ghost arm reads
        # this; the solid arm reads the live snapshot. Populated on the
        # first successful snapshot refresh and on every jog/home.
        self._target_joints: list[float] | None = None
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
        # Per-joint commanded velocity (rad/s) used by the accel-limited
        # PD tracker. Always reset to zero on (re)seed so the tracker
        # never carries pre-jog/pre-estop velocity into a new motion.
        self._stream_vel: list[float] | None = None
        # 1€ filter for the IK'd joint vector. Runs upstream of the
        # history buffer (and therefore upstream of the PD tracker), so
        # it cleans Quest jitter before the streaming loop ever sees it.
        # Reset whenever we re-seed the buffer from live joints —
        # otherwise the first post-mode-switch sample fights the
        # filter's stale history and produces a visible jump.
        self._joint_filter = OneEuroFilterND(
            6,
            min_cutoff=_JOINT_FILTER_MIN_CUTOFF_HZ,
            beta=_JOINT_FILTER_BETA,
            d_cutoff=_JOINT_FILTER_D_CUTOFF_HZ,
        )
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
        self._joint_poll_thread: threading.Thread | None = None
        self._slow_poll_thread: threading.Thread | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        log.info("connecting to UF850 at %s …", self._arm_ip)
        self._driver.connect()
        log.info("UF850 connected")
        self._refresh_snapshot()
        self._joint_poll_thread = threading.Thread(
            target=self._joint_poll_loop, daemon=True, name="xarm-joint-poll"
        )
        self._slow_poll_thread = threading.Thread(
            target=self._slow_poll_loop, daemon=True, name="xarm-slow-poll"
        )
        self._joint_poll_thread.start()
        self._slow_poll_thread.start()
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
            # Same logical state — but if we're meant to be driving and the real
            # servo path is actually down, re-assert rather than silently no-op.
            # "Down" = the stream thread died, OR it's paused, OR the controller
            # has fallen out of servo mode 1 (the post-home mode-0 latch). Keying
            # off the REAL controller mode (not just thread liveness) lets the
            # per-step _set_real_arm(True) recover a dead step; otherwise the
            # policy moves only the gripper while the joints sit frozen.
            if new_val:
                api = getattr(self._driver, "_api", None)
                real_mode = self._read_mode(api) if api is not None else None
                if (not self._stream_alive()) or self._stream_paused.is_set() or (real_mode not in (1, None)):
                    log.warning(
                        "xarm: drive_real_arm on but servo path down "
                        "(stream_alive=%s paused=%s mode=%s); re-asserting servo mode",
                        self._stream_alive(), self._stream_paused.is_set(), real_mode,
                    )
                    self._set_streaming_mode(True)
            return
        self._drive_real_arm = new_val
        self._set_streaming_mode(new_val)

    def _stream_alive(self) -> bool:
        """True only when the 250 Hz servo-stream thread is actually running —
        i.e. the real arm will follow joint targets, not just the gripper."""
        t = self._stream_thread
        return bool(self._streaming_active and t is not None and t.is_alive())

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
            # Drop any stale stream thread first so re-asserting (desync repair)
            # never spawns a second one. No-op when none is running.
            self._stop_stream_thread()
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
        seed_t = time.perf_counter()
        with self._stream_target_lock:
            self._stream_history.clear()
            self._stream_history.append((seed_t, list(cur_rad)))
            self._stream_cmd = list(cur_rad)
            self._stream_vel = [0.0] * 6
            self._joint_filter.reset(x0=list(cur_rad), t0=seed_t)
            # Fresh drive starts in Quest-hold mode (no stale policy target);
            # a policy POST flips _policy_drive_active on and supplies a target.
            self._policy_drive_active = False
            self._policy_target = None
            self._policy_start = None
            self._policy_target_t = None
            self._policy_interval_s = _POLICY_INTERP_NOMINAL_S
            self._policy_last_arrival = None
        # Reset the gripper dedupe so a new policy run's FIRST gripper command
        # is never swallowed by the previous run's last-commanded position.
        self._last_gripper_sdk_pos = None
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
        """Fixed-rate 250 Hz loop: render the buffered ghost trajectory
        at (now − _STREAM_DELAY_S) via linear interpolation, push it.

        This is the *only* place set_servo_angle_j gets called. The
        command IS the (interpolated) buffered joint state — no
        smoothing, no chase. The real arm physically traces the ghost's
        path at the ghost's speed.
        """
        api = getattr(self._driver, "_api", None)
        if api is None:
            return
        next_tick = time.perf_counter()
        while not self._stream_stop.is_set():
            if not self._stream_paused.is_set():
                if self._policy_drive_active:
                    # ── FAITHFUL + SMOOTH POLICY PATH ─────────────────────────
                    # Linearly interpolate from _policy_start toward the latest
                    # _policy_target across _policy_interval_s, emitting a fresh
                    # setpoint every 250 Hz tick. This upsamples the policy's
                    # ~30 Hz waypoints into smooth ≥60 Hz motion that still passes
                    # through each waypoint exactly — NO 1€ filter, NO Catmull-Rom,
                    # NO PD chase (re-applying the training-time smoothing would
                    # double-lag the arm; that was the bug we removed).
                    now = time.perf_counter()
                    with self._stream_target_lock:
                        tgt = (
                            list(self._policy_target)
                            if self._policy_target is not None
                            else None
                        )
                        start = (
                            list(self._policy_start)
                            if self._policy_start is not None
                            else None
                        )
                        t0 = self._policy_target_t
                        horizon = self._policy_interval_s
                    if tgt is not None and len(tgt) == 6:
                        if start is not None and len(start) == 6 and t0 is not None:
                            h = min(
                                _POLICY_INTERP_CLAMP_MAX_S,
                                max(_POLICY_INTERP_CLAMP_MIN_S, horizon),
                            )
                            alpha = (now - t0) / h
                            if alpha < 0.0:
                                alpha = 0.0
                            elif alpha > 1.0:
                                alpha = 1.0
                            out = [
                                start[i] + alpha * (tgt[i] - start[i])
                                for i in range(6)
                            ]
                        else:
                            out = tgt
                        try:
                            with self._lock:
                                if (
                                    not self._stream_paused.is_set()
                                    and self._streaming_active
                                ):
                                    api.set_servo_angle_j(angles=out, is_radian=True)
                        except Exception as exc:
                            log.debug("policy interp set_servo_angle_j failed: %s", exc)
                        with self._stream_target_lock:
                            self._stream_cmd = out
                            self._stream_vel = [0.0] * 6
                else:
                    # ── QUEST TELEOP PATH (unchanged) ─────────────────────────
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
                        # Acceleration-limited critically-damped PD tracker.
                        # Smooths jittery Quest hand input; _stream_vel carries
                        # the per-joint commanded velocity across cycles.
                        kp = _STREAM_KP
                        kd = _STREAM_KD
                        vel_state = (
                            list(self._stream_vel)
                            if self._stream_vel is not None
                            else [0.0] * 6
                        )
                        new_cmd = [0.0] * 6
                        new_vel = [0.0] * 6
                        for i in range(6):
                            err = target[i] - cmd[i]
                            a = (
                                kp * err
                                - kd * vel_state[i]
                            )
                            if a > _STREAM_MAX_ACCEL_RAD_S2:
                                a = _STREAM_MAX_ACCEL_RAD_S2
                            elif a < -_STREAM_MAX_ACCEL_RAD_S2:
                                a = -_STREAM_MAX_ACCEL_RAD_S2
                            v = vel_state[i] + a * _STREAM_PERIOD
                            if v > _STREAM_MAX_VEL_RAD_S:
                                v = _STREAM_MAX_VEL_RAD_S
                            elif v < -_STREAM_MAX_VEL_RAD_S:
                                v = -_STREAM_MAX_VEL_RAD_S
                            new_vel[i] = v
                            new_cmd[i] = cmd[i] + v * _STREAM_PERIOD
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
                            self._stream_vel = new_vel
            next_tick += _STREAM_PERIOD
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                # Behind schedule (poll loop probably held the SDK
                # lock). Reset rather than fire a catch-up burst that
                # would only feed back into jitter.
                next_tick = time.perf_counter()

    def _sample_history_locked(self, t: float) -> list[float] | None:
        """Catmull-Rom interpolation of the buffered ghost trajectory at
        time ``t``. Caller must hold ``_stream_target_lock``.

        Each interior segment is a cubic Hermite with endpoint tangents
        derived from the neighbor samples — adjacent segments share
        velocity at the joining sample, so the rendered trajectory has
        no slope discontinuities at Quest-frame boundaries. (Bare
        linear interpolation, the previous behavior, had a velocity
        step at every sample, which the operator felt as a stop-start
        rhythm at the cadence of Quest's frame rate.)

        Edge cases:
        * Empty buffer → None.
        * Single entry → hold at it.
        * t before oldest entry → hold at oldest (stream warmup).
        * t after newest entry → forward-extrapolate using the last
          segment's slope for up to ``_STREAM_EXTRAP_S``, capped at
          one segment span of overshoot. Lets the arm coast through
          brief Quest input gaps instead of stopping at the newest
          sample and re-accelerating when the next one lands.
        * Fewer than 4 points → fall back to linear (Catmull-Rom
          needs four control points).
        """
        h = self._stream_history
        if not h:
            return None
        cutoff = t - _STREAM_HISTORY_WINDOW_S
        # Keep at least 4 samples so the spline always has neighbors.
        while len(h) > 4 and h[1][0] < cutoff:
            h.popleft()
        n = len(h)
        if n == 1:
            return list(h[0][1])
        if t <= h[0][0]:
            return list(h[0][1])
        if t >= h[-1][0]:
            if n >= 2 and (t - h[-1][0]) < _STREAM_EXTRAP_S:
                t0, j0 = h[-2]
                t1, j1 = h[-1]
                span = t1 - t0
                if span > 0.0:
                    # Cap overshoot at one full segment span — bounds
                    # the worst case if the operator actually stopped
                    # between the last two samples.
                    alpha = min((t - t1) / span, 1.0)
                    return [j1[k] + alpha * (j1[k] - j0[k]) for k in range(6)]
            return list(h[-1][1])
        # Find segment containing t.
        for i in range(n - 1):
            t0, j0 = h[i]
            t1, j1 = h[i + 1]
            if t0 <= t <= t1:
                span = t1 - t0
                if span <= 0.0:
                    return list(j1)
                alpha = (t - t0) / span
                if n < 4:
                    return [j0[k] + alpha * (j1[k] - j0[k]) for k in range(6)]
                # Neighbors for the cubic; clamp to endpoints at edges
                # so the spline degenerates gracefully rather than
                # reaching outside the buffer.
                p0 = h[i - 1][1] if i > 0 else j0
                p3 = h[i + 2][1] if (i + 2) < n else j1
                return _catmull_rom6(p0, j0, j1, p3, alpha)
        return list(h[-1][1])

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
            # The xArm only accepts a mode switch from a STOPPED state. If a prior
            # position-mode move is still slewing, set_mode(1) is silently dropped
            # and the arm stays in mode 0 — joints frozen, gripper still works
            # (the every-other-step "only the claw moves" latch). Wait for the
            # controller to settle, then switch and VERIFY, retrying until it
            # actually reports mode 1.
            self._wait_until_stopped(api, timeout=2.0)
            switched = False
            for attempt in range(4):
                try:
                    api.motion_enable(enable=True)
                    api.set_mode(1)
                    api.set_state(0)
                except Exception as exc:
                    log.warning("xarm: restore set_mode(1) attempt %d failed: %s", attempt + 1, exc)
                    time.sleep(0.06)
                    continue
                mode = self._read_mode(api)
                if mode is None or mode == 1:   # None = SDK can't report; trust the call
                    switched = True
                    break
                log.warning("xarm: set_mode(1) did not take (mode=%s); settling + retrying", mode)
                self._wait_until_stopped(api, timeout=1.0)
                time.sleep(0.08)
            if switched:
                log.info("xarm: restored servo mode (1)")
            else:
                log.error("xarm: FAILED to restore servo mode after retries — joints will not drive")
            # Re-seed the stream from live joints AND reset the policy-drive
            # anchors, so the first post-restore tick doesn't replay the previous
            # step's stale target through a clamped lerp.
            try:
                cur_deg = self._driver.read_joint_state()
                cur_rad = [math.radians(j) for j in cur_deg[:6]]
                seed_t = time.perf_counter()
                with self._stream_target_lock:
                    self._stream_cmd = list(cur_rad)
                    self._stream_history.clear()
                    self._stream_history.append((seed_t, list(cur_rad)))
                    self._stream_vel = [0.0] * 6
                    self._joint_filter.reset(x0=list(cur_rad), t0=seed_t)
                    self._policy_drive_active = False
                    self._policy_target = None
                    self._policy_start = None
                    self._policy_target_t = None
                    self._policy_interval_s = _POLICY_INTERP_NOMINAL_S
                    self._policy_last_arrival = None
            except Exception:
                pass
        finally:
            self._stream_paused.clear()

    def _wait_until_stopped(self, api: Any, *, timeout: float) -> None:
        """Block (bounded) until the controller leaves the moving state, so a
        mode switch will be honored. xArm get_state(): 1 = in-motion (SPORT);
        anything else (2 READY / 3 PAUSE / 4 STOP) counts as settled."""
        deadline = time.perf_counter() + max(0.0, timeout)
        while time.perf_counter() < deadline:
            try:
                code, state = api.get_state()
            except Exception:
                return
            if code != 0 or state != 1:
                return
            time.sleep(0.03)

    @staticmethod
    def _read_mode(api: Any) -> int | None:
        """Best-effort read of the live controller mode (1 = servo-joint).
        Returns None when the SDK build doesn't expose it (then we trust the
        set_mode call instead of looping forever)."""
        try:
            getm = getattr(api, "get_mode", None)
            if callable(getm):
                r = getm()
                if isinstance(r, (tuple, list)) and len(r) == 2:
                    return int(r[1])
                return int(r)
            m = getattr(api, "mode", None)
            return int(m) if m is not None else None
        except Exception:
            return None

    def get_joint_filter_params(self) -> dict[str, float]:
        """Read the current 1€ joint-filter params (min_cutoff Hz, beta)."""
        with self._stream_target_lock:
            return self._joint_filter.get_params()

    def set_joint_filter_params(
        self,
        *,
        min_cutoff: float | None = None,
        beta: float | None = None,
    ) -> dict[str, float]:
        """Hot-update the 1€ joint filter from the dashboard. Lock-held
        so a Quest sample being filtered mid-call can't read torn
        values, and the filter's running state survives the change."""
        with self._stream_target_lock:
            self._joint_filter.set_params(min_cutoff=min_cutoff, beta=beta)
            return self._joint_filter.get_params()

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
                target_joints, _ = self._ik_for_pose(target_pose)
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
                    mvacc=DEFAULT_JOINT_ACCEL_RAD_S2,
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
        """IK the given TCP pose and store the resulting joints as
        ``_target_joints`` (drives the dashboard ghost arm) and as a
        new entry in the stream history (drives the real arm when
        ``drive_real_arm`` is on). Whatever branch the SDK IK picks is
        what we use — the operator's hand motion is smooth enough in
        practice that we don't need to filter out the rare branch
        switches."""
        # Soft rate-cap at 200 Hz — protects the xArm SDK from being
        # hammered if Unity ever runs hotter than the network round-trip.
        now = time.perf_counter()
        last = getattr(self, "_last_ghost_t", 0.0)
        if now - last < 1.0 / 200.0:
            return {"throttled": True}
        self._last_ghost_t = now

        # IK rejection here used to short-circuit and return ``ik_fail``,
        # which the bridge counted toward an auto-reanchor watchdog. That
        # path produced a teleop "freeze" the operator could only escape
        # by toggling drive mode off/on. Now: ``_ik_for_pose`` already
        # falls back to the last-known joints on rejection, so we just
        # use that and keep pushing samples — the arm holds smoothly at
        # the last reachable pose and resumes the moment IK accepts a
        # new pose again. No frozen state, no manual recovery needed.
        target_joints, _ik_ok = self._ik_for_pose(pose_mm_deg)
        with self._lock:
            self._target_joints = target_joints
            self._last_snapshot["target_joints"] = list(target_joints)

        # Append to the streaming thread's replay buffer. The loop will
        # render this (interpolated) _STREAM_DELAY_S from now. See
        # _stream_loop / _sample_history_locked.
        #
        # The joint vector goes through the 1€ filter first — its job
        # is to knock the per-Quest-frame jitter out of the trajectory
        # before the renderer commits to replaying it.
        if self._drive_real_arm and not self._estopped:
            sample_t = time.perf_counter()
            with self._stream_target_lock:
                # Quest path: hand input + IK noise needs filtering;
                # Quest-tuned PD provides the "ease-in" coast.
                self._policy_drive_active = False
                filtered = self._joint_filter(sample_t, target_joints)
                self._stream_history.append((sample_t, list(filtered)))
        return {"target_joints": list(target_joints), "pose": list(pose_mm_deg)}

    def set_joint_target(
        self,
        joints_rad: list[float] | tuple[float, ...],
        *,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        """Sibling of ``set_ghost_target_pose`` that skips IK — the caller
        already has joint targets (e.g. from a learned policy that
        outputs joint deltas). Same downstream plumbing: updates the
        ghost arm's ``_target_joints``, and when ``drive_real_arm`` is on
        pushes the joints through the 1€ filter into the stream history
        so the 250 Hz tracker replays them onto the real arm.

        ``gripper`` is the LeRobot ``gripper_pos`` convention (0 = fully
        open, 1 = fully closed). Forwarded straight to the SDK's
        ``set_gripper_position`` (continuous 0–850 scale) so a policy
        can command intermediate jaw widths."""
        if self._estopped:
            return {"error": "estopped"}
        joints = list(joints_rad)
        if len(joints) != 6:
            return {"error": f"joints must have length 6, got {len(joints)}"}
        try:
            joints = [float(j) for j in joints]
        except (TypeError, ValueError):
            return {"error": "joints must be numeric"}

        with self._lock:
            self._target_joints = list(joints)
            self._last_snapshot["target_joints"] = list(joints)

        applied_gripper: float | None = None
        if gripper is not None:
            try:
                g = max(0.0, min(1.0, float(gripper)))
            except (TypeError, ValueError):
                return {"error": "gripper must be numeric in [0, 1]"}
            applied_gripper = g
            # Mirror the snapshot's "gripper" Literal so the dashboard's
            # state pill matches what the policy commanded.
            self._last_snapshot["gripper"] = "closed" if g > 0.5 else "open"
            self._last_snapshot["gripper_pos"] = g
            api = getattr(self._driver, "_api", None)
            if api is not None and hasattr(api, "set_gripper_position"):
                # SDK convention: 0 = closed, 850 = open. Our 0–1
                # follows the snapshot's (0=open, 1=closed), so invert.
                sdk_pos = int(round((1.0 - g) * 850.0))
                # Dedupe: only re-issue the SDK call if the commanded
                # position moved by ≥ 40 (≈ 5 % of full travel = 4.7 mm
                # jaw movement). Otherwise we're re-targeting the
                # gripper controller faster than it can move, which
                # interferes with its own close cycle.
                last = self._last_gripper_sdk_pos
                if last is None or abs(sdk_pos - last) >= 40:
                    self._last_gripper_sdk_pos = sdk_pos
                    try:
                        with self._lock:
                            api.set_gripper_position(sdk_pos, wait=False)
                    except Exception as exc:
                        log.debug("set_gripper_position(%s) failed: %s", sdk_pos, exc)

        if self._drive_real_arm and not self._estopped:
            # FAITHFUL + SMOOTH policy actuation. Store the model's RAW absolute
            # target plus the interpolation anchors the 250 Hz stream loop needs
            # to lerp toward it. NO 1€ filter, NO Catmull-Rom, NO PD chase —
            # those already shaped the achieved-state training labels (the
            # recorder saves encoder reads and action[t]=state[t+1]), so
            # re-filtering/re-chasing the prediction double-lags the arm below
            # the demonstrated trajectory. Quest's set_ghost_target_pose keeps
            # the full smoothing path unchanged.
            now = time.perf_counter()
            with self._stream_target_lock:
                # Anchor the lerp at where the arm is currently commanded so the
                # motion is continuous across waypoints. _stream_cmd is seeded
                # from live joints in _start_stream_thread, so it is always a
                # valid current position (Quest hold, prior policy tick, or seed).
                if self._stream_cmd is not None and len(self._stream_cmd) == 6:
                    self._policy_start = list(self._stream_cmd)
                else:
                    self._policy_start = list(joints)
                # Adapt the lerp horizon to the policy's actual cadence so it
                # finishes right as the next target lands (no dwell, no lunge),
                # but ignore chunk-boundary inference stalls so they don't bias
                # the steady-state tracking.
                if self._policy_last_arrival is not None:
                    gap = now - self._policy_last_arrival
                    if _POLICY_INTERP_MIN_S <= gap <= _POLICY_INTERP_EMA_MAX_S:
                        self._policy_interval_s = (
                            (1.0 - _POLICY_INTERP_EMA_ALPHA) * self._policy_interval_s
                            + _POLICY_INTERP_EMA_ALPHA * gap
                        )
                self._policy_last_arrival = now
                self._policy_target_t = now
                self._policy_target = list(joints)
                self._policy_drive_active = True
        return {
            "target_joints": list(joints),
            "applied_gripper": applied_gripper,
            "drive_real_arm": bool(self._drive_real_arm),
        }

    def home(self) -> dict[str, Any]:
        self._require_armed()
        with self._lock:
            # home() uses set_servo_angle (position mode), which only
            # works in mode 0. If we're currently streaming (mode 1 for
            # Quest teleop), drop briefly to 0 and restore afterwards.
            restore = self._ensure_position_mode()
            try:
                # Joint-space home — bypass IK so we always land in the
                # canonical "gripper forward, jaws upright" branch. Going
                # through TCP-space IK silently picks a wrist-flipped
                # branch (joint 4 = 180°) that reaches the same TCP but
                # rolls the gripper 180° about its pointing axis, which
                # reads as "facing backward" to the operator.
                target_joints_rad = [
                    math.radians(j) for j in DEFAULT_HOME_JOINTS_850_DEG
                ]
                self._target_joints = list(target_joints_rad)
                api = getattr(self._driver, "_api", None)
                if api is None:
                    raise RuntimeError("xArm SDK not connected")
                # wait=True: block until the position-mode slew finishes. The
                # xArm only honors a mode switch from a STOPPED state, so the
                # finally's _restore_servo_mode must not run while we're still
                # slewing — otherwise set_mode(1) is silently dropped and the arm
                # stays in mode 0 (joints frozen, only the gripper moves). This
                # is the every-other-step "no motion" latch.
                code = api.set_servo_angle(
                    angle=list(DEFAULT_HOME_JOINTS_850_DEG),
                    speed=HOMING_JOINT_SPEED_RAD_S,
                    mvacc=HOMING_JOINT_ACCEL_RAD_S2,
                    is_radian=False,
                    wait=True,
                )
                if code != 0:
                    raise RuntimeError(f"xArm home returned code {code}")
                snap = self._refresh_snapshot_locked()
            finally:
                if restore:
                    self._restore_servo_mode()
            return snap

    def _ik_for_pose(
        self, pose_mm_deg: tuple[float, ...]
    ) -> tuple[list[float], bool]:
        """Query the xArm's onboard IK. Returns ``(joints, ok)``: on
        failure ``joints`` falls back to the last known target so the
        caller can keep going. The ``ok`` flag is preserved for
        diagnostic logging but no longer drives any state machine —
        the auto-reanchor watchdog that used to consume it has been
        removed (it caused a "freeze until you toggle drive mode"
        UX bug). SDK call is one TCP roundtrip; serialized via ``_lock``."""
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
        # The parallel gripper runs on its own SDK channel
        # (set_gripper_mode / set_gripper_enable / set_gripper_position)
        # and is mechanically independent of the arm's servo mode.
        # Earlier this path flipped arm mode 1→0→1 around the SDK call
        # "to be safe", which paused the stream thread and re-seeded
        # both the history buffer and the 1€ filter from live joints —
        # so squeezing the gripper mid-motion erased the in-flight
        # Quest trajectory and produced a visible hitch on the other
        # joints. Holding _lock is enough to serialize the SDK socket
        # with the streaming thread; mode stays put.
        with self._lock:
            # wait=False so the poll thread can refresh gripper
            # position (now read from the SDK) while the gripper is
            # mid-travel — the rendered fingers in the dashboard
            # animate live.
            self._driver.set_gripper(state, wait=False)
            # NOTE: deliberately NOT calling _refresh_snapshot_locked
            # here. The three SDK reads it does (~15 ms lock-hold)
            # were the dominant latency between the operator's grip
            # button and the gripper hardware starting to move. The
            # background poll thread will pick up the new gripper
            # position on its next tick (20 Hz, so ≤50 ms behind).
            self._last_snapshot["gripper"] = state
            snap = dict(self._last_snapshot)
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
                seed_t = time.perf_counter()
                with self._stream_target_lock:
                    self._stream_cmd = list(cur_rad)
                    self._stream_history.clear()
                    self._stream_history.append((seed_t, list(cur_rad)))
                    self._stream_vel = [0.0] * 6
                    self._joint_filter.reset(x0=list(cur_rad), t0=seed_t)
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

    def _joint_poll_loop(self) -> None:
        """250 Hz: just joint_state. Matches the xArm controller's
        internal mode-1 cycle — polling faster than this just returns
        the same value. Mutates only the joints/t fields of
        _last_snapshot so the slow poll's tcp_pose / gripper_pos
        updates don't get clobbered. Each read is one TCP roundtrip
        (~1–2 ms) and is serialized via _lock with the stream thread."""
        while not self._poll_stop.is_set():
            try:
                with self._lock:
                    joints_deg = self._driver.read_joint_state()
                joints_rad = [math.radians(j) for j in joints_deg[:6]]
                with self._lock:
                    if self._target_joints is None:
                        self._target_joints = list(joints_rad)
                    self._last_snapshot["joints"] = joints_rad
                    self._last_snapshot["t"] = time.time()
            except Exception:
                pass
            self._poll_stop.wait(0.004)  # 250 Hz

    def _slow_poll_loop(self) -> None:
        """10 Hz: tcp_pose + gripper_position. Both change slowly enough
        that pushing faster just burns SDK socket time the streaming
        thread and IK queries need. Each iteration mutates fields, not
        the whole dict, so it cooperates with the fast joint poller."""
        while not self._poll_stop.is_set():
            try:
                with self._lock:
                    pose_mm_deg = self._driver.read_tcp_pose()
                with self._lock:
                    raw_grip = self._driver.read_gripper_position()
                tcp_pos_mm = [
                    float(pose_mm_deg[0]),
                    float(pose_mm_deg[1]),
                    float(pose_mm_deg[2]),
                ]
                tcp_rpy = [math.radians(float(v)) for v in pose_mm_deg[3:6]]
                grip01: float | None
                if not math.isnan(raw_grip):
                    grip01 = max(0.0, min(1.0, 1.0 - raw_grip / 850.0))
                else:
                    grip01 = None
                with self._lock:
                    self._last_snapshot["tcp_pos_mm"] = tcp_pos_mm
                    self._last_snapshot["tcp_rpy"] = tcp_rpy
                    if grip01 is not None:
                        self._last_snapshot["gripper_pos"] = grip01
            except Exception:
                pass
            self._poll_stop.wait(0.1)  # 10 Hz


