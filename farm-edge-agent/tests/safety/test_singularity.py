from dataclasses import dataclass, field

from farm_edge_agent.safety import Pose
from farm_edge_agent.safety.singularity import SingularityCheck


@dataclass
class FakeDriver:
    unreachable: set[tuple[float, ...]] = field(default_factory=set)
    colliding: set[tuple[float, ...]] = field(default_factory=set)

    def check_pose_reachable(self, pose: Pose) -> bool:
        return tuple(pose) not in self.unreachable

    def check_self_collision(self, pose: Pose) -> bool:
        return tuple(pose) in self.colliding


def test_reachable_collision_free_pose_passes() -> None:
    check = SingularityCheck(FakeDriver())
    assert check.check(Pose(0.1, 0.0, 0.2)).ok


def test_unreachable_pose_rejected() -> None:
    bad = Pose(1.5, 0.0, 0.0)
    driver = FakeDriver(unreachable={tuple(bad)})
    result = SingularityCheck(driver).check(bad)
    assert not result.ok
    assert result.event is not None
    assert result.event.detail["reason"] == "unreachable"


def test_self_collision_pose_rejected() -> None:
    bad = Pose(0.0, 0.0, -0.05)
    driver = FakeDriver(colliding={tuple(bad)})
    result = SingularityCheck(driver).check(bad)
    assert not result.ok
    assert result.event is not None
    assert result.event.detail["reason"] == "self_collision"


def test_check_short_circuits_before_collision_call() -> None:
    """Unreachable poses must not also be sent through the collision check."""

    calls: list[str] = []

    class Recorder:
        def check_pose_reachable(self, pose: Pose) -> bool:
            calls.append("reach")
            return False

        def check_self_collision(self, pose: Pose) -> bool:
            calls.append("collide")
            return False

    SingularityCheck(Recorder()).check(Pose(0, 0, 0))
    assert calls == ["reach"]
