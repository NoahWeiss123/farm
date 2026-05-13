from __future__ import annotations

import base64
import json

import pytest
from farm_edge_agent.protocol.messages import (
    Ack,
    ActionChunk,
    Control,
    EePoseDelta,
    Hello,
    ObsChunk,
    SafetyEvent,
    TcpPose,
    parse_message,
)
from farm_shared.errors import ErrorCode


def _round_trip(msg):
    raw = msg.model_dump_json()
    return parse_message(raw)


def test_hello_round_trip() -> None:
    msg = Hello(
        protocol_version="1.2.0",
        agent_version="0.0.1",
        feature_flags={"ghosting": True},
    )
    back = _round_trip(msg)
    assert isinstance(back, Hello)
    assert back == msg


def test_ack_round_trip_accepted() -> None:
    msg = Ack(protocol_version="1.2.0", accepted=True)
    back = _round_trip(msg)
    assert isinstance(back, Ack)
    assert back.accepted is True
    assert back.reason is None


def test_ack_round_trip_rejected_with_reason() -> None:
    msg = Ack(protocol_version="2.0.0", accepted=False, reason="major mismatch")
    back = _round_trip(msg)
    assert isinstance(back, Ack)
    assert back.accepted is False
    assert back.reason == "major mismatch"


def test_obs_chunk_round_trip_with_uri_frames() -> None:
    msg = ObsChunk(
        run_id="r_1",
        ts=1715000000.0,
        frames={"wrist": "r2://run/r_1/wrist/000.jpg"},
        joint_state=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        tcp_pose=TcpPose(x=0.3, y=0.0, z=0.4, roll=0.0, pitch=0.0, yaw=0.0),
        gripper_state="open",
    )
    back = _round_trip(msg)
    assert isinstance(back, ObsChunk)
    assert back == msg


def test_obs_chunk_with_inline_bytes_frame_serializes_as_base64() -> None:
    payload = b"\x89PNG\r\n\x1a\n"
    msg = ObsChunk(
        run_id="r_2",
        ts=0.0,
        frames={"wrist": payload},
        joint_state=[0.0] * 6,
        tcp_pose=TcpPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),
        gripper_state="closed",
    )
    raw = msg.model_dump_json()
    on_wire = json.loads(raw)
    assert on_wire["frames"]["wrist"] == base64.b64encode(payload).decode("ascii")


def test_action_chunk_round_trip() -> None:
    msg = ActionChunk(
        run_id="r_3",
        chunk_id=7,
        actions=[
            EePoseDelta(dx=0.01, gripper=None),
            EePoseDelta(dz=-0.005, gripper="close"),
        ],
        suggested_dwell_ms=250,
    )
    back = _round_trip(msg)
    assert isinstance(back, ActionChunk)
    assert back == msg


def test_safety_event_serializes_code_as_enum_name() -> None:
    msg = SafetyEvent(run_id="r_4", ts=42.0, code=ErrorCode.E3001, halted=True)
    raw = msg.model_dump_json()
    on_wire = json.loads(raw)
    assert on_wire["code"] == "E3001"
    back = parse_message(raw)
    assert isinstance(back, SafetyEvent)
    assert back.code is ErrorCode.E3001
    assert back.halted is True


def test_control_round_trip_all_commands() -> None:
    for command in ("pause", "resume", "abort", "home"):
        msg = Control(run_id="r_5", command=command)  # type: ignore[arg-type]
        back = _round_trip(msg)
        assert isinstance(back, Control)
        assert back.command == command


def test_parse_message_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        parse_message('{"type": "nope"}')


def test_parse_message_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        parse_message("[1, 2, 3]")
