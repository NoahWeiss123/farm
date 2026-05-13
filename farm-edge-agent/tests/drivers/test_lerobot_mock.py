from __future__ import annotations

from pathlib import Path

from farm_edge_agent.drivers.base import Pose
from farm_edge_agent.drivers.lerobot_mock import (
    HOME_POSE,
    LerobotMockDriver,
)

FIXTURE = Path(__file__).parent / "fixtures" / "canned_trajectory.jsonl"

TARGET_A: Pose = (400.0, 0.0, 300.0, 0.0, 0.0, 0.0)
TARGET_B: Pose = (400.0, 50.0, 250.0, 0.0, 0.0, 0.0)


def _run_canned_scenario(driver: LerobotMockDriver) -> None:
    driver.connect()
    driver.move_to(TARGET_A, velocity_cap=50.0)
    driver.set_gripper("closed")
    driver.move_to(TARGET_B, velocity_cap=50.0)
    driver.home()
    driver.disconnect()


def test_connect_disconnect_clean() -> None:
    driver = LerobotMockDriver()
    driver.connect()
    assert driver._connected is True
    driver.disconnect()
    assert driver._connected is False


def test_move_to_reaches_target_within_1mm() -> None:
    driver = LerobotMockDriver()
    target: Pose = (450.0, 25.0, 275.0, 0.0, 0.0, 0.0)
    driver.move_to(target, velocity_cap=50.0)
    final = driver.read_tcp_pose()
    for i in range(3):
        assert abs(final[i] - target[i]) < 1.0


def test_home_returns_to_canonical_pose() -> None:
    driver = LerobotMockDriver()
    driver.move_to(TARGET_B, velocity_cap=50.0)
    driver.home()
    assert driver.read_tcp_pose() == HOME_POSE
    assert driver.gripper_state == "open"


def test_trajectory_log_matches_fixture(tmp_path: Path) -> None:
    log = tmp_path / "trajectory.jsonl"
    driver = LerobotMockDriver(seed=42, trajectory_log_path=log)
    _run_canned_scenario(driver)
    assert log.read_bytes() == FIXTURE.read_bytes()


def test_same_seed_produces_identical_logs(tmp_path: Path) -> None:
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    _run_canned_scenario(LerobotMockDriver(seed=7, trajectory_log_path=log_a))
    _run_canned_scenario(LerobotMockDriver(seed=7, trajectory_log_path=log_b))
    assert log_a.read_bytes() == log_b.read_bytes()


def test_set_gripper_closed_then_read_reports_closed() -> None:
    driver = LerobotMockDriver()
    driver.set_gripper("closed")
    assert driver.gripper_state == "closed"


def test_is_estop_armed_by_default() -> None:
    driver = LerobotMockDriver()
    assert driver.is_estop_armed() is True


def test_joint_state_has_six_values() -> None:
    driver = LerobotMockDriver()
    assert len(driver.read_joint_state()) == 6
    driver.move_to(TARGET_A, velocity_cap=50.0)
    assert len(driver.read_joint_state()) == 6
