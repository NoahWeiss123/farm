"""Edge Agent safety boundary.

Composes envelope, velocity, watchdog, singularity, e-stop, and calibration
checks. The control loop talks to a single `SafetyEnforcer`; nothing else
reaches the arm without passing every gate here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

SafetyEventKind = Literal[
    "envelope_violation",
    "velocity_clamp",
    "watchdog_timeout",
    "singularity_rejected",
    "estop_not_armed",
    "calibration_stale",
]

Severity = Literal["warning", "violation"]


class Pose(NamedTuple):
    """6-DoF Cartesian pose in the arm base frame. Meters and radians."""

    x: float
    y: float
    z: float
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0


@dataclass(frozen=True)
class SafetyEvent:
    """A single safety check firing. Maps 1:1 to a run-record event."""

    kind: SafetyEventKind
    severity: Severity
    message: str
    code: str
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    """Pass/fail outcome of a single safety check."""

    ok: bool
    event: SafetyEvent | None = None

    @classmethod
    def pass_(cls) -> CheckResult:
        return cls(ok=True, event=None)

    @classmethod
    def fail(cls, event: SafetyEvent) -> CheckResult:
        return cls(ok=False, event=event)


@dataclass
class ActionChunk:
    """Slice of trajectory the control loop wants to send to the arm.

    Joint values in radians, TCP waypoints in meters. Either may be empty;
    a chunk with only joint targets skips TCP checks and vice versa.
    """

    joint_positions: Sequence[Sequence[float]] = field(default_factory=list)
    joint_velocities: Sequence[Sequence[float]] = field(default_factory=list)
    tcp_waypoints: Sequence[Pose] = field(default_factory=list)
    duration_s: float = 0.0


__all__ = [
    "ActionChunk",
    "CheckResult",
    "Pose",
    "SafetyEvent",
    "SafetyEventKind",
    "Severity",
]
