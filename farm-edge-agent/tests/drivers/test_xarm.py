"""XArmDriver unit tests. Mocks _xarm_sdk_shim.XArmAPI; never hits real hardware."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from farm_edge_agent.drivers._xarm_sdk_shim import XArmAPI, XArmSDKMissingError
from farm_edge_agent.drivers.xarm import (
    DEFAULT_HOME_POSE_850,
    XArmDriver,
    XArmDriverError,
)

_XARM_SDK_AVAILABLE = importlib.util.find_spec("xarm") is not None

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "xarm_motion_log.jsonl"


class RecordingXArm:
    """Mock xArm SDK that records every call. Returns canned, deterministic values."""

    _DEFAULTS: dict[str, Any] = {
        "motion_enable": 0,
        "set_mode": 0,
        "set_state": 0,
        "set_position": 0,
        "set_gripper_position": 0,
        "set_gripper_mode": 0,
        "set_gripper_enable": 0,
        "set_gripper_speed": 0,
        "get_position": (0, [300.0, 0.0, 300.0, 180.0, 0.0, 0.0]),
        "get_servo_angle": (0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "get_state": (0, 1),
        "disconnect": None,
    }

    def __init__(self, returns: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._returns = dict(self._DEFAULTS)
        if returns:
            self._returns.update(returns)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "calls":
            raise AttributeError(name)
        return self._make_method(name)

    def _make_method(self, name: str):
        def method(*args: Any, **kwargs: Any) -> Any:
            ret = self._returns.get(name)
            self.calls.append(
                {
                    "method": name,
                    "args": list(args),
                    "kwargs": dict(kwargs),
                    "returns": _to_jsonable(ret),
                }
            )
            return ret

        return method


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


@pytest.fixture
def recording_arm(monkeypatch: pytest.MonkeyPatch) -> RecordingXArm:
    arm = RecordingXArm()
    monkeypatch.setattr(
        "farm_edge_agent.drivers._xarm_sdk_shim.XArmAPI",
        lambda ip, **kwargs: arm,
    )
    return arm


def _patch_arm(monkeypatch: pytest.MonkeyPatch, arm: Any) -> None:
    monkeypatch.setattr(
        "farm_edge_agent.drivers._xarm_sdk_shim.XArmAPI",
        lambda ip, **kwargs: arm,
    )


def test_connect_enables_motion_then_sets_mode_and_state(
    recording_arm: RecordingXArm,
) -> None:
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    methods = [c["method"] for c in recording_arm.calls]
    assert methods == [
        "motion_enable",
        "set_mode",
        "set_state",
        "set_gripper_mode",
        "set_gripper_enable",
        "set_gripper_speed",
    ]
    assert recording_arm.calls[0]["kwargs"] == {"enable": True}
    assert recording_arm.calls[1]["args"] == [0]
    assert recording_arm.calls[2]["args"] == [0]
    assert recording_arm.calls[3]["args"] == [0]
    assert recording_arm.calls[4]["args"] == [True]


def test_move_to_reads_current_pose_then_sends_relative_set_position(
    recording_arm: RecordingXArm,
) -> None:
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    recording_arm.calls.clear()
    driver.move_to((10.0, 0.0, 0.0, 0.0, 0.0, 0.0), velocity_cap=75.0)
    methods = [c["method"] for c in recording_arm.calls]
    assert methods == ["get_position", "set_position"]
    set_pos = recording_arm.calls[1]
    assert set_pos["args"] == [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert set_pos["kwargs"]["relative"] is True
    assert set_pos["kwargs"]["wait"] is True
    assert set_pos["kwargs"]["speed"] == 75.0


def test_move_to_out_of_envelope_raises_before_sdk_set_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = RecordingXArm(returns={"get_position": (0, [500.0, 0.0, 400.0, 0.0, 0.0, 0.0])})
    _patch_arm(monkeypatch, arm)
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    arm.calls.clear()
    with pytest.raises(XArmDriverError) as exc:
        driver.move_to((200.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert exc.value.code == "FARM-E3001"
    methods = [c["method"] for c in arm.calls]
    assert methods == ["get_position"]
    assert "set_position" not in methods


def test_move_to_in_envelope_with_custom_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = RecordingXArm(returns={"get_position": (0, [50.0, 0.0, 50.0, 0.0, 0.0, 0.0])})
    _patch_arm(monkeypatch, arm)
    driver = XArmDriver(
        "192.168.1.10",
        envelope_min=(0.0, -100.0, 0.0),
        envelope_max=(100.0, 100.0, 100.0),
    )
    driver.connect()
    arm.calls.clear()
    driver.move_to((10.0, 0.0, 10.0, 0.0, 0.0, 0.0))
    methods = [c["method"] for c in arm.calls]
    assert "set_position" in methods


def test_sdk_timeout_return_code_becomes_structured_farm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = RecordingXArm(returns={"set_position": 9})
    _patch_arm(monkeypatch, arm)
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    with pytest.raises(XArmDriverError) as exc:
        driver.move_to((10.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert exc.value.code == "FARM-E1005"
    assert "timed out" in str(exc.value).lower()


def test_sdk_raised_exception_wrapped_as_structured_farm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = MagicMock()
    arm.motion_enable.side_effect = RuntimeError("connection refused")
    _patch_arm(monkeypatch, arm)
    driver = XArmDriver("192.168.1.10")
    with pytest.raises(XArmDriverError) as exc:
        driver.connect()
    assert exc.value.code == "FARM-E1009"
    assert "RuntimeError" in str(exc.value)


def test_sdk_nonzero_return_code_wrapped_as_structured_farm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = RecordingXArm(returns={"motion_enable": 1})
    _patch_arm(monkeypatch, arm)
    driver = XArmDriver("192.168.1.10")
    with pytest.raises(XArmDriverError) as exc:
        driver.connect()
    assert exc.value.code == "FARM-E1009"


def test_is_estop_armed_true_for_sport_state(recording_arm: RecordingXArm) -> None:
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    assert driver.is_estop_armed() is True


def test_is_estop_armed_false_for_estop_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = RecordingXArm(returns={"get_state": (0, 5)})
    _patch_arm(monkeypatch, arm)
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    assert driver.is_estop_armed() is False


def test_home_issues_absolute_set_position_to_home_pose(
    recording_arm: RecordingXArm,
) -> None:
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    recording_arm.calls.clear()
    driver.home()
    set_pos_calls = [c for c in recording_arm.calls if c["method"] == "set_position"]
    assert len(set_pos_calls) == 1
    assert set_pos_calls[0]["args"] == list(DEFAULT_HOME_POSE_850)
    assert set_pos_calls[0]["kwargs"]["relative"] is False


def test_home_pose_override_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = RecordingXArm()
    _patch_arm(monkeypatch, arm)
    custom_home = (250.0, 0.0, 250.0, 180.0, 0.0, 0.0)
    driver = XArmDriver("192.168.1.10", home_pose=custom_home)
    driver.connect()
    arm.calls.clear()
    driver.home()
    set_pos = next(c for c in arm.calls if c["method"] == "set_position")
    assert tuple(set_pos["args"]) == custom_home


def test_set_gripper_maps_each_state_to_a_position(
    recording_arm: RecordingXArm,
) -> None:
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    recording_arm.calls.clear()
    driver.set_gripper("open")
    driver.set_gripper("closed")
    driver.set_gripper("grasping")
    positions = [
        c["args"][0] for c in recording_arm.calls if c["method"] == "set_gripper_position"
    ]
    assert positions == [850, 0, 400]


def test_read_tcp_pose_returns_sdk_pose_as_tuple(
    recording_arm: RecordingXArm,
) -> None:
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    pose = driver.read_tcp_pose()
    assert pose == (300.0, 0.0, 300.0, 180.0, 0.0, 0.0)


def test_read_joint_state_returns_list_of_floats(
    recording_arm: RecordingXArm,
) -> None:
    driver = XArmDriver("192.168.1.10")
    driver.connect()
    joints = driver.read_joint_state()
    assert joints == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_disconnect_is_a_noop_before_connect() -> None:
    driver = XArmDriver("192.168.1.10")
    driver.disconnect()


def test_method_calls_before_connect_raise_structured_error() -> None:
    driver = XArmDriver("192.168.1.10")
    with pytest.raises(XArmDriverError) as exc:
        driver.read_tcp_pose()
    assert exc.value.code == "FARM-E1009"


@pytest.mark.skipif(
    _XARM_SDK_AVAILABLE,
    reason="xarm SDK is installed; shim does not need to raise",
)
def test_shim_raises_structured_error_when_sdk_not_installed() -> None:
    with pytest.raises(XArmSDKMissingError) as exc:
        XArmAPI("192.168.1.10")
    assert exc.value.code == "FARM-E1010"
    assert "pip install" in str(exc.value)


def test_canned_sequence_matches_golden_motion_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = RecordingXArm()
    _patch_arm(monkeypatch, arm)

    driver = XArmDriver("192.168.1.10")
    driver.connect()
    driver.move_to((10.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    driver.set_gripper("closed")
    driver.home()
    driver.disconnect()

    expected = [
        json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()
    ]
    assert arm.calls == expected
