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

        # Routing: only the right-hand pose is wired to motion today. The
        # other topics are accepted (and shape-checked) so the Quest client
        # connects cleanly, but their semantics are deferred until the new
        # Unity app lands. Left-hand pose, twists, inputs all flow through
        # here and can be hooked up later without re-touching the wire.
        if topic == "/q2r_right_hand_pose":
            self._on_quest_right_hand(msg)
        elif topic == "/teleop_data_collector/episode_event":
            log.info("ros-tcp episode_event: %s", msg.data)
        elif topic == "/uf850/real_control_enable":
            log.info("ros-tcp real_control_enable: %s", msg.data)

    def _on_quest_right_hand(self, msg: messages.PoseStamped) -> Any:
        # Stub: log the pose for now. The actual Quest → arm mapping needs
        # the re-anchor scheme from quest_teleop_node.py; that lives outside
        # this rebuild's scope ("ports ready", not "policy wired").
        log.debug(
            "ros-tcp right_hand_pose: pos=(%.3f, %.3f, %.3f)",
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        )

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


__all__ = ["RosTcpBridge"]
