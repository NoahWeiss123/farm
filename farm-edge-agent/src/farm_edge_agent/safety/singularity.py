"""IK reachability + self-collision check delegated to the arm driver."""

from __future__ import annotations

from typing import Protocol

from . import CheckResult, Pose, SafetyEvent


class SupportsReachability(Protocol):
    """Minimum driver surface the singularity check requires.

    Implemented by the xArm driver (task 012) and the mock driver. Kept here
    as a Protocol so this module has no compile-time dependency on either.
    """

    def check_pose_reachable(self, pose: Pose) -> bool: ...
    def check_self_collision(self, pose: Pose) -> bool: ...


class SingularityCheck:
    """Reject a waypoint if it is unreachable or would self-collide.

    Both checks are delegated to the driver, which wraps the xArm SDK's IK +
    collision routines. Failure produces a `SafetyEvent` rather than raising.
    """

    def __init__(self, driver: SupportsReachability) -> None:
        self._driver = driver

    def check(self, pose: Pose) -> CheckResult:
        if not self._driver.check_pose_reachable(pose):
            return CheckResult.fail(
                SafetyEvent(
                    kind="singularity_rejected",
                    severity="violation",
                    code="FARM-E3003",
                    message="commanded pose is unreachable (IK has no solution)",
                    detail={"pose": list(pose), "reason": "unreachable"},
                )
            )
        if self._driver.check_self_collision(pose):
            return CheckResult.fail(
                SafetyEvent(
                    kind="singularity_rejected",
                    severity="violation",
                    code="FARM-E3003",
                    message="commanded pose triggers a self-collision",
                    detail={"pose": list(pose), "reason": "self_collision"},
                )
            )
        return CheckResult.pass_()
