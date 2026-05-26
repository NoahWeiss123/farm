"""Wire-format round-trip tests for the ROS-TCP bridge.

Exercises both the frame layer (``read_frame``/``write_frame``) and the
message schemas (PoseStamped, OVR2ROSInputs, JointState, …) end-to-end via
a real TCP socket pair.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import closing

import pytest
from farm_edge_agent.ros_bridge import RosTcpBridge, messages
from farm_edge_agent.ros_bridge.wire import (
    Reader,
    Writer,
    read_frame,
    write_frame,
)


def test_writer_reader_primitive_roundtrip() -> None:
    w = Writer()
    w.bool(True)
    w.int32(-42)
    w.uint32(7)
    w.float32(0.5)
    w.float64(1.25)
    w.string("hello")
    w.float64_array([1.0, -2.5, 3.14])
    w.string_array(["a", "bb"])

    r = Reader(w.to_bytes())
    assert r.bool() is True
    assert r.int32() == -42
    assert r.uint32() == 7
    assert abs(r.float32() - 0.5) < 1e-6
    assert r.float64() == 1.25
    assert r.string() == "hello"
    assert r.float64_array() == [1.0, -2.5, 3.14]
    assert r.string_array() == ["a", "bb"]
    assert r.remaining() == 0


def test_pose_stamped_roundtrip() -> None:
    src = messages.PoseStamped(
        header=messages.Header(stamp=messages.Time(sec=12, nsec=345), frame_id="world"),
        pose=messages.Pose(
            position=messages.Point(x=0.1, y=-0.2, z=0.3),
            orientation=messages.Quaternion(x=0.0, y=0.707, z=0.0, w=0.707),
        ),
    )
    body = messages.encode(src)
    out = messages.decode(messages.PoseStamped, body)
    assert out.header.stamp.sec == 12
    assert out.header.stamp.nsec == 345
    assert out.header.frame_id == "world"
    assert out.pose.position.x == pytest.approx(0.1)
    assert out.pose.position.y == pytest.approx(-0.2)
    assert out.pose.orientation.y == pytest.approx(0.707)


def test_ovr2ros_inputs_roundtrip() -> None:
    src = messages.OVR2ROSInputs(
        button_upper=True, button_lower=False,
        thumb_stick_horizontal=0.25, thumb_stick_vertical=-0.5,
        press_index=0.9, press_middle=0.1,
    )
    out = messages.decode(messages.OVR2ROSInputs, messages.encode(src))
    assert out.button_upper is True
    assert out.button_lower is False
    assert out.thumb_stick_horizontal == pytest.approx(0.25)
    assert out.press_index == pytest.approx(0.9)


def test_joint_state_roundtrip() -> None:
    src = messages.JointState(
        header=messages.Header(stamp=messages.Time(sec=1, nsec=2), frame_id="base"),
        name=["joint1", "joint2"],
        position=[0.5, -0.5],
        velocity=[],
        effort=[],
    )
    out = messages.decode(messages.JointState, messages.encode(src))
    assert out.name == ["joint1", "joint2"]
    assert out.position == [0.5, -0.5]
    assert out.header.frame_id == "base"


# ── Frame / TCP smoke test ──────────────────────────────────────────────────


def test_frame_roundtrip_over_socket() -> None:
    """A real socket pair must round-trip the (topic, body) framing."""
    a, b = socket.socketpair()
    with closing(a), closing(b):
        body = b"hello-body"
        write_frame(a, "/v1/test", body)
        topic, got = read_frame(b)
    assert topic == "/v1/test"
    assert got == body


class _FakeSupervisor:
    def snapshot(self) -> dict:
        return {
            "joints": [0.0, -0.5, -0.5, 0.0, -1.57, 0.0],
            "tcp_pos_mm": [0.0, -700.0, 250.0],
            "tcp_rpy": [3.14, 0.0, 0.0],
            "gripper": "open",
            "gripper_pos": 0.0,
            "t": 1.0,
        }


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_bridge_publishes_joint_state_to_connected_client() -> None:
    port = _free_port()
    bridge = RosTcpBridge(_FakeSupervisor(), port=port, publish_hz=20.0)  # type: ignore[arg-type]
    bridge.start()
    try:
        # Give the accept loop a moment to be ready.
        time.sleep(0.05)
        with closing(socket.create_connection(("127.0.0.1", port), timeout=2.0)) as c:
            c.settimeout(2.0)
            # The publisher pumps at 20 Hz; we should see /joint_states quickly.
            topic, body = read_frame(c)
            assert topic == "/joint_states", f"expected /joint_states, got {topic!r}"
            msg = messages.decode(messages.JointState, body)
            assert msg.name == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            assert len(msg.position) == 6
    finally:
        bridge.stop()


def test_bridge_routes_inbound_quest_pose_without_crashing() -> None:
    """Quest-pose frames sent from a client should be parsed silently."""
    port = _free_port()
    received_event = threading.Event()

    class _SpySupervisor(_FakeSupervisor):
        pass

    bridge = RosTcpBridge(_SpySupervisor(), port=port, publish_hz=5.0)  # type: ignore[arg-type]
    bridge.start()
    try:
        time.sleep(0.05)
        with closing(socket.create_connection(("127.0.0.1", port), timeout=2.0)) as c:
            ps = messages.PoseStamped()
            ps.pose.position.x = 1.0
            write_frame(c, "/q2r_right_hand_pose", messages.encode(ps))
            # Give the bridge thread a moment to drain.
            time.sleep(0.1)
        # The bridge must not have crashed — connecting again proves the
        # listener is still up.
        with closing(socket.create_connection(("127.0.0.1", port), timeout=2.0)):
            received_event.set()
        assert received_event.is_set()
    finally:
        bridge.stop()
