"""Workspace envelope: axis-aligned bounding box in Cartesian space."""

from __future__ import annotations

from collections.abc import Callable

from . import CheckResult, Pose, SafetyEvent

OnEvent = Callable[[SafetyEvent], None]


class Envelope:
    """Reject poses outside a configured AABB in the arm base frame.

    A pose exactly on the boundary is accepted; one micron outside is rejected.
    """

    def __init__(
        self,
        min_xyz: tuple[float, float, float],
        max_xyz: tuple[float, float, float],
        on_event: OnEvent | None = None,
    ) -> None:
        for axis, lo, hi in zip("xyz", min_xyz, max_xyz, strict=True):
            if lo > hi:
                raise ValueError(f"envelope {axis} min {lo} > max {hi}")
        self._min = min_xyz
        self._max = max_xyz
        self._on_event = on_event

    def check(self, pose: Pose) -> CheckResult:
        position = (pose.x, pose.y, pose.z)
        for axis, value, lo, hi in zip(
            "xyz", position, self._min, self._max, strict=True
        ):
            if value < lo or value > hi:
                event = SafetyEvent(
                    kind="envelope_violation",
                    severity="violation",
                    code="FARM-E3001",
                    message=(
                        f"commanded pose outside workspace on {axis}-axis: "
                        f"{value:.4f} not in [{lo:.4f}, {hi:.4f}]"
                    ),
                    detail={
                        "axis": axis,
                        "value": value,
                        "min": lo,
                        "max": hi,
                        "pose": list(pose),
                    },
                )
                if self._on_event is not None:
                    self._on_event(event)
                return CheckResult.fail(event)
        return CheckResult.pass_()
