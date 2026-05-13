"""Tests for RunState.from_current."""

from __future__ import annotations

from dataclasses import dataclass, field

from farm_edge_agent.recovery import GripperState, Pose
from farm_edge_agent.recovery.handoff import RunState

HOME_POSE: Pose = (300.0, 0.0, 300.0, 0.0, 0.0, 0.0)
CURRENT_POSE: Pose = (240.0, 10.0, 220.0, 0.0, 0.0, 0.0)


@dataclass
class StubDriver:
    tcp_pose: Pose = CURRENT_POSE
    joints: list[float] = field(
        default_factory=lambda: [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
    )
    gripper: GripperState = "grasping"

    def move_to(self, pose: Pose, velocity_cap: float) -> None:  # pragma: no cover
        raise AssertionError("from_current should not move the arm")

    def set_gripper(self, state: GripperState) -> None:  # pragma: no cover
        raise AssertionError("from_current should not actuate the gripper")

    def read_tcp_pose(self) -> Pose:
        return self.tcp_pose

    def read_joint_state(self) -> list[float]:
        return list(self.joints)

    def read_gripper_state(self) -> GripperState:
        return self.gripper


@dataclass
class StubSafety:
    home_pose: Pose = HOME_POSE
    velocity_cap: float = 75.0
    watchdog_armed: bool = True

    def clamp_to_envelope(self, pose: Pose) -> Pose:  # pragma: no cover
        return pose

    def disarm_watchdog(self) -> None:  # pragma: no cover
        self.watchdog_armed = False


PLAN = {
    "plan_id": "p-1",
    "nodes": [{"id": "n0", "skill": "pick"}, {"id": "n1", "skill": "place"}],
}


def test_from_current_captures_live_driver_state() -> None:
    driver = StubDriver()
    safety = StubSafety()

    state = RunState.from_current(
        driver, safety, run_id="r-42", plan=PLAN, last_chunk=3
    )

    assert state.run_id == "r-42"
    assert state.joint_state == list(driver.joints)
    assert state.tcp_pose == CURRENT_POSE
    assert state.gripper_state == "grasping"
    assert state.plan == PLAN
    assert state.last_chunk_index == 3
    assert state.task_progress_index == 0
    assert state.home_pose == HOME_POSE
    assert state.velocity_cap == 75.0
    assert state.observation == {}
    assert state.critic_summary is None


def test_from_current_propagates_optional_progress_observation_and_summary() -> None:
    driver = StubDriver()
    safety = StubSafety()
    obs = {"wrist": "frame://w/7", "overhead": "frame://o/7"}

    state = RunState.from_current(
        driver,
        safety,
        run_id="r-42",
        plan=PLAN,
        last_chunk=5,
        task_progress_index=1,
        observation=obs,
        critic_summary="grasp slipped during retreat",
    )

    assert state.last_chunk_index == 5
    assert state.task_progress_index == 1
    assert state.observation == obs
    assert state.critic_summary == "grasp slipped during retreat"


def test_from_current_snapshot_is_frozen_and_decoupled() -> None:
    driver = StubDriver()
    safety = StubSafety()

    state = RunState.from_current(
        driver, safety, run_id="r-1", plan=PLAN, last_chunk=0,
        observation={"wrist": "frame://w/0"},
    )

    # Mutating the source driver after capture should not change the snapshot.
    driver.joints[0] = 9.9
    driver.tcp_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert state.joint_state[0] != 9.9
    assert state.tcp_pose == CURRENT_POSE


def test_from_current_with_empty_plan_is_allowed() -> None:
    driver = StubDriver()
    safety = StubSafety()

    state = RunState.from_current(
        driver, safety, run_id="r-empty", plan={}, last_chunk=-1
    )

    assert state.plan == {}
    assert state.last_chunk_index == -1
