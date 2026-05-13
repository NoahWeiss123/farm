"""Tests for the five recovery primitives and the Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from farm_edge_agent.recovery import (
    AbortedRunError,
    GripperState,
    Pose,
    RecoveryEvent,
)
from farm_edge_agent.recovery.primitives import (
    Registry,
    abort_safely,
    home,
    open_gripper,
    relocalize,
    retry_grasp,
)

HOME_POSE: Pose = (300.0, 0.0, 300.0, 0.0, 0.0, 0.0)
SAFETY_CAP = 75.0
START_POSE: Pose = (250.0, 50.0, 200.0, 0.0, 0.0, 0.0)
GRASP_POSE: Pose = (280.0, 30.0, 180.0, 0.0, 0.0, 0.0)


@dataclass
class FakeDriver:
    """Records every call so tests can assert order and arguments."""

    tcp_pose: Pose = START_POSE
    joint_state: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    gripper_state: GripperState = "closed"
    calls: list[tuple[str, tuple, dict]] = field(default_factory=list)

    def move_to(self, pose: Pose, velocity_cap: float) -> None:
        self.calls.append(("move_to", (pose, velocity_cap), {}))
        self.tcp_pose = pose

    def set_gripper(self, state: GripperState) -> None:
        self.calls.append(("set_gripper", (state,), {}))
        self.gripper_state = state

    def read_tcp_pose(self) -> Pose:
        self.calls.append(("read_tcp_pose", (), {}))
        return self.tcp_pose

    def read_joint_state(self) -> list[float]:
        self.calls.append(("read_joint_state", (), {}))
        return list(self.joint_state)

    def read_gripper_state(self) -> GripperState:
        self.calls.append(("read_gripper_state", (), {}))
        return self.gripper_state


@dataclass
class FakeSafety:
    """Test stub that records disarm and exposes a known envelope clamp."""

    home_pose: Pose = HOME_POSE
    velocity_cap: float = SAFETY_CAP
    watchdog_armed: bool = True
    clamp_to: Pose | None = None
    clamp_calls: list[Pose] = field(default_factory=list)
    disarm_calls: int = 0

    def clamp_to_envelope(self, pose: Pose) -> Pose:
        self.clamp_calls.append(pose)
        return self.clamp_to if self.clamp_to is not None else pose

    def disarm_watchdog(self) -> None:
        self.disarm_calls += 1
        self.watchdog_armed = False


@dataclass
class FakePerception:
    """Returns a deterministic frame dict."""

    payload: dict[str, Any] = field(
        default_factory=lambda: {"wrist": "frame://wrist/0", "overhead": None}
    )
    captures: int = 0

    def capture(self) -> dict[str, Any]:
        self.captures += 1
        return dict(self.payload)


def _call_names(driver: FakeDriver) -> list[str]:
    return [call[0] for call in driver.calls]


def test_home_moves_to_safety_home_then_opens_gripper() -> None:
    driver = FakeDriver()
    safety = FakeSafety()
    events: list[RecoveryEvent] = []

    result = home(driver, safety, sink=events.append)

    assert result.ok
    assert result.primitive == "home"
    assert _call_names(driver) == ["move_to", "set_gripper"]
    assert driver.calls[0] == ("move_to", (HOME_POSE, SAFETY_CAP), {})
    assert driver.calls[1] == ("set_gripper", ("open",), {})
    assert events == [RecoveryEvent(primitive="home", detail={})]


def test_home_uses_safety_velocity_cap_not_a_hardcoded_value() -> None:
    driver = FakeDriver()
    safety = FakeSafety(velocity_cap=12.5)

    home(driver, safety)

    pose_arg, vel_arg = driver.calls[0][1]
    assert pose_arg == HOME_POSE
    assert vel_arg == 12.5


def test_open_gripper_sets_open_and_emits_event() -> None:
    driver = FakeDriver(gripper_state="closed")
    safety = FakeSafety()
    events: list[RecoveryEvent] = []

    result = open_gripper(driver, safety, sink=events.append)

    assert result.ok
    assert driver.gripper_state == "open"
    assert _call_names(driver) == ["set_gripper"]
    assert events == [RecoveryEvent(primitive="open_gripper", detail={})]


def test_relocalize_captures_frames_and_reads_pose() -> None:
    driver = FakeDriver()
    perception = FakePerception()
    events: list[RecoveryEvent] = []

    result = relocalize(driver, perception, sink=events.append)

    assert result.ok
    assert perception.captures == 1
    assert _call_names(driver) == ["read_tcp_pose", "read_joint_state"]
    assert result.detail["frames"] == perception.payload
    assert result.detail["tcp_pose"] == list(START_POSE)
    assert result.detail["joint_state"] == list(driver.joint_state)
    assert len(events) == 1
    assert events[0].primitive == "relocalize"


def test_retry_grasp_moves_to_last_tcp_then_closes_gripper() -> None:
    driver = FakeDriver()
    safety = FakeSafety()
    events: list[RecoveryEvent] = []

    result = retry_grasp(driver, safety, GRASP_POSE, sink=events.append)

    assert result.ok
    assert _call_names(driver) == ["move_to", "set_gripper"]
    assert driver.calls[0] == ("move_to", (GRASP_POSE, SAFETY_CAP), {})
    assert driver.calls[1] == ("set_gripper", ("closed",), {})
    assert events[0].primitive == "retry_grasp"
    assert events[0].detail == {"last_tcp": list(GRASP_POSE)}


def test_retry_grasp_respects_velocity_cap_from_safety() -> None:
    driver = FakeDriver()
    safety = FakeSafety(velocity_cap=4.0)

    retry_grasp(driver, safety, GRASP_POSE)

    _, vel_arg = driver.calls[0][1]
    assert vel_arg == 4.0


def test_abort_safely_clamps_descends_opens_and_disarms() -> None:
    safe_target: Pose = (260.0, 0.0, 60.0, 0.0, 0.0, 0.0)
    driver = FakeDriver(tcp_pose=(800.0, 800.0, 800.0, 0.0, 0.0, 0.0))
    safety = FakeSafety(clamp_to=safe_target)
    events: list[RecoveryEvent] = []

    result = abort_safely(driver, safety, sink=events.append)

    assert result.ok
    assert _call_names(driver) == ["read_tcp_pose", "move_to", "set_gripper"]
    assert driver.calls[1] == ("move_to", (safe_target, SAFETY_CAP), {})
    assert driver.calls[2] == ("set_gripper", ("open",), {})
    assert safety.clamp_calls == [(800.0, 800.0, 800.0, 0.0, 0.0, 0.0)]
    assert safety.disarm_calls == 1
    assert safety.watchdog_armed is False
    assert events[0].primitive == "abort_safely"
    assert events[0].detail == {"safe_pose": list(safe_target)}


def test_abort_safely_second_call_raises() -> None:
    driver = FakeDriver()
    safety = FakeSafety()

    abort_safely(driver, safety)

    with pytest.raises(AbortedRunError):
        abort_safely(driver, safety)
    assert safety.disarm_calls == 1


def test_no_sink_is_a_silent_no_op() -> None:
    driver = FakeDriver()
    safety = FakeSafety()

    home(driver, safety)
    open_gripper(driver, safety)


def test_all_motion_primitives_pass_safety_velocity_cap() -> None:
    driver = FakeDriver()
    safety = FakeSafety(velocity_cap=33.3, clamp_to=(280.0, 0.0, 60.0, 0.0, 0.0, 0.0))

    home(driver, safety)
    retry_grasp(driver, safety, GRASP_POSE)
    abort_safely(driver, safety)

    move_calls = [c for c in driver.calls if c[0] == "move_to"]
    assert len(move_calls) == 3
    for _, args, _ in move_calls:
        _, vel = args
        assert vel == 33.3


def test_registry_default_lookup_returns_each_primitive() -> None:
    registry = Registry()

    assert registry.get("home") is home
    assert registry.get("open_gripper") is open_gripper
    assert registry.get("relocalize") is relocalize
    assert registry.get("retry_grasp") is retry_grasp
    assert registry.get("abort_safely") is abort_safely


def test_registry_unknown_name_raises_key_error() -> None:
    registry = Registry()

    with pytest.raises(KeyError):
        registry.get("teleport")


def test_registry_register_overrides_existing_primitive() -> None:
    registry = Registry()
    sentinel = object()

    def custom(driver, safety, **kwargs):  # type: ignore[no-untyped-def]
        return sentinel  # pragma: no cover

    registry.register("home", custom)
    assert registry.get("home") is custom


def test_registry_names_exposes_keys() -> None:
    registry = Registry()
    assert set(registry.names()) == {
        "home",
        "open_gripper",
        "relocalize",
        "retry_grasp",
        "abort_safely",
    }
