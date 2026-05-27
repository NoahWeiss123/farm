"""TCP listener that speaks the Unity ROS-TCP-Endpoint wire format.

Accepts connections on the configured port (default ``10000``), reads
``(topic, body)`` frames, dispatches incoming Quest messages to the sim,
and pumps ``/joint_states`` back to every connected peer at a fixed rate
so the in-headset HUD bars stay live.

This is intentionally a subset of the upstream ROS-TCP-Endpoint:
* No service-call routing yet — ``/uf850/reset_pose`` is dispatched as a
  fire-and-forget on the same channel.
* No ``__handshake`` config exchange — Quest's ``ROSConnection`` simply
  starts publishing the moment its TCP connection succeeds, so the bridge
  accepts that flow today. The handshake can be added when needed.
"""

from __future__ import annotations

import logging
import math
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

from . import messages
from .wire import read_frame, write_frame

if TYPE_CHECKING:
    from farm_edge_agent.server.supervisor import Supervisor

log = logging.getLogger("farm.ros_bridge")

_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# Auto re-anchor after this many consecutive rejected/ik_fail ghost
# updates. At ~80 Hz Quest input that's ~0.3 s of "the ghost wasn't
# moving" — clearly a stall, but quick enough that the operator's hand
# hasn't drifted far from where the re-snap will land.
_AUTO_REANCHOR_STALLS = 25


class RosTcpBridge:
    """One TCP listener, fan-out publisher, fan-in routing.

    Designed so the Quest VR client can connect unmodified using
    ``Unity.Robotics.ROSTCPConnector`` once it exists. Until then, the
    Python smoke test in ``tests/ros_bridge/test_wire.py`` exercises the
    same frame paths.
    """

    def __init__(
        self,
        supervisor: Supervisor,
        *,
        host: str = "127.0.0.1",
        port: int = 10000,
        publish_hz: float = 10.0,
    ) -> None:
        self._supervisor = supervisor
        self._host = host
        self._port = port
        self._publish_period = 1.0 / max(0.1, float(publish_hz))

        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._clients_lock = threading.Lock()
        self._clients: list[socket.socket] = []
        self._accept_thread: threading.Thread | None = None
        self._publish_thread: threading.Thread | None = None

        # Trigger-gated anchor scheme (Quest right hand). Real arm motion
        # gated on RIGHT TRIGGER held. On rising edge, snapshot both the
        # controller pose AND the arm TCP pose — from then on, controller
        # deltas (translation + rotation) apply to the arm anchor until
        # the trigger releases. Ghost freezes on release; pressing again
        # picks a fresh anchor wherever the arm currently is.
        self._trigger_held = False
        self._anchor_armed = False
        # Rising-edge bookkeeping for the Quest's command inputs:
        #   right A button (button_lower) → start / save recording
        #   right B button (button_upper) → cancel recording (only while
        #                                   recording, no-op otherwise)
        #   right grip squeeze (press_middle) → toggle gripper open/closed
        #   right stick click (thumb_stick_click) → toggle drive_real_arm
        #     (just digital ↔ digital + real)
        self._a_held = False
        self._b_held = False
        self._grip_held = False
        self._stick_click_held = False
        self._anchor_ctrl_pos: tuple[float, float, float] | None = None
        self._anchor_ctrl_quat: tuple[float, float, float, float] | None = None
        self._anchor_arm_xyz_mm: tuple[float, float, float] | None = None
        self._anchor_arm_quat: tuple[float, float, float, float] | None = None
        # Most-recent Quest right-hand pose in arm-base mm/rad — published
        # in the snapshot so the dashboard can render a gizmo where the
        # user's hand is. Set whenever a pose arrives (trigger-gated or not).
        self._last_ctrl_xyz_mm: tuple[float, float, float] | None = None
        self._last_ctrl_rpy: tuple[float, float, float] | None = None
        self._last_ctrl_t: float = 0.0
        # Per-trigger-press viz anchors: when the trigger goes down, we
        # snap the dashboard gizmo to the arm's current TCP pose and
        # then drive it by the controller delta. Both clear on
        # release; on the next press they capture fresh anchors.
        self._viz_ctrl_anchor: tuple[float, float, float] | None = None
        self._viz_ctrl_anchor_quat: tuple[float, float, float, float] | None = None
        self._viz_arm_anchor_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._viz_arm_anchor_quat: tuple[float, float, float, float] | None = None
        # Auto re-anchor on persistent ghost stalls. The arm-anchor is
        # captured at trigger DOWN, then targets = anchor + Δcontroller.
        # As the operator's hand drifts away from the anchor, the
        # computed target eventually leaves the reachable workspace and
        # IK fails every frame — symptom: ghost AND real arm both
        # freeze until the operator manually release+represses. We count
        # consecutive rejected/ik_fail returns from set_ghost_target_pose
        # and force a fresh anchor capture once the streak crosses
        # _AUTO_REANCHOR_STALLS. Equivalent to release+repress but
        # automatic. _ms threshold is "long enough to be a real stall,
        # short enough that the operator barely notices the re-snap."
        self._ghost_stall_streak = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self._host, self._port))
        s.listen(4)
        s.settimeout(0.5)
        self._sock = s
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()
        log.info("ros-tcp bridge listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.close()
                except Exception:
                    pass
            self._clients.clear()

    @property
    def port(self) -> int:
        return self._port

    @property
    def last_controller_pose(self) -> dict | None:
        """Most-recent Quest right-controller pose as a snapshot-shaped
        dict, or None if no pose has been received yet. The supervisor
        merges this into every world snapshot so the dashboard's gizmo
        works regardless of which backend is active."""
        if self._last_ctrl_xyz_mm is None or self._last_ctrl_rpy is None:
            return None
        return {
            "xyz_mm": list(self._last_ctrl_xyz_mm),
            "rpy": list(self._last_ctrl_rpy),
            "trigger_held": self._trigger_held,
            "t": self._last_ctrl_t,
        }

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    # ── accept + per-client read loop ───────────────────────────────────────

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            # On macOS the conn socket can inherit the 0.5 s timeout of
            # the listen socket — that makes read_frame raise
            # socket.timeout on any frame gap > 0.5 s. Force it back to
            # blocking so we wait however long the Quest needs.
            conn.settimeout(None)
            # Quest pose frames are small (~80 B body + header). Without
            # TCP_NODELAY, Nagle waits for an ACK before sending the
            # next one, which on the wire path between headset and
            # laptop can stall an entire pose update by 40 ms and feed
            # straight into arm-motion jitter.
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError as exc:
                log.debug("TCP_NODELAY set failed on %s: %s", addr, exc)
            log.info("ros-tcp client connected: %s", addr)
            with self._clients_lock:
                self._clients.append(conn)
            t = threading.Thread(target=self._client_loop, args=(conn, addr), daemon=True)
            t.start()

    def _client_loop(self, conn: socket.socket, addr) -> None:
        try:
            while not self._stop.is_set():
                topic, body = read_frame(conn)
                self._dispatch(topic, body)
        except (ConnectionError, ValueError, OSError) as exc:
            log.info("ros-tcp client %s disconnected: %s", addr, exc)
        finally:
            with self._clients_lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            try:
                conn.close()
            except Exception:
                pass

    def _dispatch(self, topic: str, body: bytes) -> None:
        schema = messages.QUEST_TOPIC_SCHEMAS.get(topic)
        if schema is None:
            log.debug("ros-tcp unknown topic dropped: %s (%d bytes)", topic, len(body))
            return
        try:
            msg = messages.decode(schema, body)
        except Exception as exc:
            log.warning("ros-tcp decode failed for %s: %s", topic, exc)
            return

        if topic == "/q2r_right_hand_pose":
            self._on_quest_right_hand(msg)
        elif topic == "/q2r_right_hand_inputs":
            self._on_right_inputs(msg)
        elif topic == "/teleop_data_collector/episode_event":
            log.info("ros-tcp episode_event: %s", msg.data)
        elif topic == "/uf850/real_control_enable":
            log.info("ros-tcp real_control_enable: %s", msg.data)

    def _on_right_inputs(self, msg: messages.OVR2ROSInputs) -> None:
        """Track the right trigger as a 'control enable' gate.

        * Rising edge → arm an anchor for the next pose frame.
        * Falling edge → clear anchor; the ghost freezes where it last was.
        """
        trigger_now = float(msg.press_index) > 0.5
        if trigger_now and not self._trigger_held:
            self._anchor_armed = True
            log.info("trigger DOWN — anchoring on next pose frame")
        elif (not trigger_now) and self._trigger_held:
            self._anchor_ctrl_pos = None
            self._anchor_ctrl_quat = None
            self._anchor_arm_xyz_mm = None
            self._anchor_arm_quat = None
            self._anchor_armed = False
            # Drop the dashboard-gizmo anchor too, so the next press
            # re-snaps the box to whatever the arm is doing at that
            # moment.
            self._viz_ctrl_anchor = None
            self._viz_ctrl_anchor_quat = None
            self._viz_arm_anchor_quat = None
            log.info("trigger UP — ghost frozen")
        self._trigger_held = trigger_now

        # ── grip (middle finger) → gripper toggle on rising edge ─────
        grip_now = float(msg.press_middle) > 0.5
        if grip_now and not self._grip_held:
            try:
                cur = self._supervisor.snapshot().get("gripper", "open")
                new = "open" if cur == "closed" else "closed"
                self._supervisor.set_gripper(new)
                log.info("grip RISING — gripper → %s", new)
            except Exception as exc:
                log.warning("gripper toggle failed: %s", exc)
        self._grip_held = grip_now

        # ── A (button_lower) → start / save recording on rising edge ──
        a_now = bool(msg.button_lower)
        if a_now and not self._a_held:
            rec = getattr(self._supervisor, "recorder", None)
            if rec is not None:
                try:
                    if rec.is_recording:
                        result = rec.stop_save()
                        log.info("A RISING — save: %s", result)
                    else:
                        result = rec.start()
                        log.info("A RISING — start: %s", result)
                except Exception as exc:
                    log.warning("recorder A press failed: %s", exc)
        self._a_held = a_now

        # ── B (button_upper) → cancel recording on rising edge ────────
        b_now = bool(msg.button_upper)
        if b_now and not self._b_held:
            rec = getattr(self._supervisor, "recorder", None)
            if rec is not None and rec.is_recording:
                try:
                    result = rec.cancel()
                    log.info("B RISING — cancel: %s", result)
                except Exception as exc:
                    log.warning("recorder B press failed: %s", exc)
        self._b_held = b_now

        # ── right stick click → toggle drive_real_arm on rising edge ──
        stick_now = bool(getattr(msg, "thumb_stick_click", False))
        if stick_now and not self._stick_click_held:
            backend = getattr(self._supervisor, "_backend", None)
            if backend is not None and hasattr(backend, "drive_real_arm"):
                backend.drive_real_arm = not backend.drive_real_arm
                log.info(
                    "stick CLICK — drive mode = %s",
                    "DIGITAL + REAL" if backend.drive_real_arm else "DIGITAL ONLY",
                )
        self._stick_click_held = stick_now

    def _on_quest_right_hand(self, msg: messages.PoseStamped) -> Any:
        """Trigger-gated relative teleop.

        While the right trigger is held, deltas between the current
        controller pose and the anchored controller pose are applied to
        the anchored arm pose. Trigger off → no motion. First frame
        after trigger press defines the anchor; subsequent frames drive
        the ghost.

        Regardless of trigger state, every pose frame updates
        ``_last_ctrl_*`` so the dashboard can render a controller gizmo.
        """
        cp = msg.pose.position
        cq = msg.pose.orientation

        # Dashboard gizmo lives only while the trigger is held. On
        # rising edge we snap it to the arm's current TCP (position +
        # orientation), then track the controller's delta from there.
        # Trigger off → no gizmo at all (rendered hidden by the
        # dashboard).
        if not self._trigger_held:
            self._last_ctrl_xyz_mm = None
            self._last_ctrl_rpy = None
            self._last_ctrl_t = time.time()
            return

        # First frame after rising edge: capture both anchors.
        if self._viz_ctrl_anchor is None:
            try:
                snap = self._supervisor.snapshot()
                self._viz_arm_anchor_mm = (
                    float(snap["tcp_pos_mm"][0]),
                    float(snap["tcp_pos_mm"][1]),
                    float(snap["tcp_pos_mm"][2]),
                )
                tcp_rpy = snap.get("tcp_rpy") or (0.0, 0.0, 0.0)
                self._viz_arm_anchor_quat = _rpy_to_quat_xyzw(
                    float(tcp_rpy[0]), float(tcp_rpy[1]), float(tcp_rpy[2])
                )
            except Exception:
                self._viz_arm_anchor_mm = (0.0, 0.0, 0.0)
                self._viz_arm_anchor_quat = (0.0, 0.0, 0.0, 1.0)
            self._viz_ctrl_anchor = (float(cp.x), float(cp.y), float(cp.z))
            self._viz_ctrl_anchor_quat = (
                float(cq.x), float(cq.y), float(cq.z), float(cq.w),
            )

        # Position: arm_anchor + (ctrl_now - ctrl_anchor)
        dx_mm = (float(cp.x) - self._viz_ctrl_anchor[0]) * 1000.0
        dy_mm = (float(cp.y) - self._viz_ctrl_anchor[1]) * 1000.0
        dz_mm = (float(cp.z) - self._viz_ctrl_anchor[2]) * 1000.0
        self._last_ctrl_xyz_mm = (
            self._viz_arm_anchor_mm[0] + dx_mm,
            self._viz_arm_anchor_mm[1] + dy_mm,
            self._viz_arm_anchor_mm[2] + dz_mm,
        )

        # Orientation: arm_anchor ⊗ (ctrl_anchor⁻¹ ⊗ ctrl_now), so on
        # trigger press the gizmo's angle matches the arm's TCP exactly,
        # then twists with the wrist.
        ctrl_now_quat = (float(cq.x), float(cq.y), float(cq.z), float(cq.w))
        if self._viz_ctrl_anchor_quat is not None and self._viz_arm_anchor_quat is not None:
            d_quat = _quat_mul(_quat_inv(self._viz_ctrl_anchor_quat), ctrl_now_quat)
            new_quat = _quat_mul(self._viz_arm_anchor_quat, d_quat)
            rx_, ry_, rz_ = _quat_to_rpy(*new_quat)
        else:
            rx_, ry_, rz_ = _quat_to_rpy(*ctrl_now_quat)
        self._last_ctrl_rpy = (rx_, ry_, rz_)
        self._last_ctrl_t = time.time()
        # Stash on the supervisor's last-snapshot so SSE pushes pick it up.
        try:
            backend = getattr(self._supervisor, "_backend", None)
            snap_holder = getattr(backend, "_last_snapshot", None)
            if isinstance(snap_holder, dict):
                snap_holder["controller_pose"] = {
                    "xyz_mm": list(self._last_ctrl_xyz_mm),
                    "rpy": list(self._last_ctrl_rpy),
                    "trigger_held": self._trigger_held,
                    "t": self._last_ctrl_t,
                }
        except Exception:
            pass

        if not self._trigger_held:
            return
        # Quest publishes meters / ROS-FLU. We store position in metres
        # (deltas stay metric until we add to the arm anchor in mm).
        ctrl_pos = (float(cp.x), float(cp.y), float(cp.z))
        ctrl_quat = (float(cq.x), float(cq.y), float(cq.z), float(cq.w))

        if self._anchor_armed:
            snap = self._supervisor.snapshot()
            try:
                tcp_mm = snap["tcp_pos_mm"]
                tcp_rpy = snap["tcp_rpy"]
            except Exception:
                return
            self._anchor_ctrl_pos = ctrl_pos
            self._anchor_ctrl_quat = ctrl_quat
            self._anchor_arm_xyz_mm = (float(tcp_mm[0]), float(tcp_mm[1]), float(tcp_mm[2]))
            self._anchor_arm_quat = _rpy_to_quat_xyzw(*tcp_rpy)
            self._anchor_armed = False
            log.info(
                "anchor set: ctrl_pos=%.3f,%.3f,%.3f  arm_mm=%.1f,%.1f,%.1f",
                *ctrl_pos, *self._anchor_arm_xyz_mm,
            )
            return  # no motion on the anchor frame itself

        if (self._anchor_ctrl_pos is None
                or self._anchor_arm_xyz_mm is None
                or self._anchor_arm_quat is None
                or self._anchor_ctrl_quat is None):
            return

        # The dashboard gizmo confirms the raw Quest pose already sits in
        # arm-base coordinates (controller moves forward = gizmo moves +X
        # in the arm frame). So delta-add directly — no axis flip.
        dx = (ctrl_pos[0] - self._anchor_ctrl_pos[0]) * 1000.0
        dy = (ctrl_pos[1] - self._anchor_ctrl_pos[1]) * 1000.0
        dz = (ctrl_pos[2] - self._anchor_ctrl_pos[2]) * 1000.0
        new_x = self._anchor_arm_xyz_mm[0] + dx
        new_y = self._anchor_arm_xyz_mm[1] + dy
        new_z = self._anchor_arm_xyz_mm[2] + dz

        # Orientation: relative rotation between current and anchor
        # controller, applied to the anchored arm pose.
        #   rel = ctrl_now · ctrl_anchor⁻¹
        #   new = rel · arm_anchor
        anc_inv = (
            -self._anchor_ctrl_quat[0], -self._anchor_ctrl_quat[1],
            -self._anchor_ctrl_quat[2], self._anchor_ctrl_quat[3],
        )
        rel = _quat_mul(ctrl_quat, anc_inv)
        new_q = _quat_mul(rel, self._anchor_arm_quat)
        rx, ry, rz = _quat_to_rpy(*new_q)

        pose = (
            float(new_x), float(new_y), float(new_z),
            math.degrees(rx), math.degrees(ry), math.degrees(rz),
        )
        try:
            result = self._supervisor.set_ghost_target_pose(pose)
        except Exception as exc:  # noqa: BLE001
            log.debug("ghost update failed: %s", exc)
            result = None
        # Auto-reanchor on persistent stalls. We don't count "throttled"
        # — that's normal at high input rates. We DO count "rejected"
        # (branch flip the unwrap couldn't collapse) and "ik_fail" (pose
        # out of reach — the dominant cause of the trigger-held freeze).
        stalled = (
            isinstance(result, dict)
            and ("rejected" in result or result.get("ik_fail"))
        )
        if stalled:
            self._ghost_stall_streak += 1
            if self._ghost_stall_streak >= _AUTO_REANCHOR_STALLS:
                reason = "ik_fail" if result.get("ik_fail") else "rejected"
                log.warning(
                    "ghost stalled %d frames (%s) — auto re-anchoring",
                    self._ghost_stall_streak,
                    reason,
                )
                # Same state machine release+repress walks through, just
                # in-place: clear both anchor sets, arm the next frame
                # to recapture. The next pose frame will snap arm-anchor
                # to the live arm TCP and ctrl-anchor to the live
                # controller pose, so the upcoming small Δcontroller
                # lands well inside the reachable workspace again.
                self._anchor_ctrl_pos = None
                self._anchor_ctrl_quat = None
                self._anchor_arm_xyz_mm = None
                self._anchor_arm_quat = None
                self._anchor_armed = True
                self._viz_ctrl_anchor = None
                self._viz_ctrl_anchor_quat = None
                self._viz_arm_anchor_quat = None
                self._ghost_stall_streak = 0
        else:
            self._ghost_stall_streak = 0

    # ── outbound publisher: /joint_states ───────────────────────────────────

    def _publish_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._broadcast_joint_state()
            except Exception as exc:
                log.warning("ros-tcp publish failed: %s", exc)
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.0, self._publish_period - elapsed))

    def _broadcast_joint_state(self) -> None:
        with self._clients_lock:
            clients = list(self._clients)
        if not clients:
            return
        snap = self._supervisor.snapshot()
        msg = messages.JointState(
            header=messages.Header(
                stamp=messages.Time(sec=int(snap["t"]), nsec=int((snap["t"] % 1) * 1e9)),
                frame_id="base_link",
            ),
            name=list(_JOINT_NAMES),
            position=list(snap["joints"]),
            velocity=[],
            effort=[],
        )
        body = messages.encode(msg)
        dead: list[socket.socket] = []
        for c in clients:
            try:
                write_frame(c, "/joint_states", body)
            except OSError:
                dead.append(c)
        if dead:
            with self._clients_lock:
                for c in dead:
                    if c in self._clients:
                        self._clients.remove(c)
                    try:
                        c.close()
                    except Exception:
                        pass


# ── quaternion helpers ─────────────────────────────────────────────────


def _quat_inv(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Conjugate of a unit quaternion (x, y, z, w) — equivalent to its
    inverse for the unit-length quats OpenXR hands us."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_mul(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Hamilton product of two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rpy_to_quat_xyzw(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Extrinsic XYZ Euler (rad) → quaternion (x, y, z, w)."""
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    return (
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    )


def _quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Quat (x, y, z, w) → extrinsic XYZ Euler (rx, ry, rz) in radians.

    Matches the xArm SDK's pose convention so the angles we feed into
    ``set_ghost_target_pose`` land on the right axes.
    """
    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (Y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    ry = math.asin(sinp)
    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)
    return rx, ry, rz


__all__ = ["RosTcpBridge"]
