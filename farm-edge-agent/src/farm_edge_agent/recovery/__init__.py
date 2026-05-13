"""Recovery primitives exposed to backends and to the Python API.

The Dispatcher and the Python API both reach for the same five primitives:
``home``, ``open_gripper``, ``relocalize``, ``retry_grasp``, ``abort_safely``.
This package is the canonical home. All motion goes through a ``Safety``
collaborator so the primitives cannot bypass the envelope or velocity cap.

The driver, safety, and perception types here are Protocols, not imports of
concrete classes. Concrete drivers (xarm, lerobot-mock) and the SafetyEnforcer
land in their own modules; they only need to structurally match these shapes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

GripperState = Literal["open", "closed", "grasping"]

Pose = tuple[float, float, float, float, float, float]
"""TCP pose in the arm's base frame: (x, y, z, rx, ry, rz). mm and radians."""


@runtime_checkable
class Driver(Protocol):
    """Subset of the arm Driver surface that the primitives talk to."""

    def move_to(self, pose: Pose, velocity_cap: float) -> None: ...

    def set_gripper(self, state: GripperState) -> None: ...

    def read_tcp_pose(self) -> Pose: ...

    def read_joint_state(self) -> list[float]: ...

    def read_gripper_state(self) -> GripperState: ...


@runtime_checkable
class Safety(Protocol):
    """Subset of the SafetyEnforcer surface the primitives depend on.

    ``home_pose`` and ``velocity_cap`` are read each call rather than baked in
    so an operator who edits config mid-session sees the new values on the next
    recovery invocation. ``clamp_to_envelope`` is what ``abort_safely`` uses to
    find the nearest in-envelope waypoint from the current TCP pose.
    """

    home_pose: Pose
    velocity_cap: float
    watchdog_armed: bool

    def clamp_to_envelope(self, pose: Pose) -> Pose: ...

    def disarm_watchdog(self) -> None: ...


@runtime_checkable
class Perception(Protocol):
    """Stub of the perception surface that ``relocalize`` calls.

    Real perception lands in its own task; the recovery layer only needs a
    callable that returns a fresh observation dict.
    """

    def capture(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RecoveryEvent:
    """Emitted by a primitive when it runs, consumed by the run record.

    Stays minimal on purpose: the Dispatcher wraps the sink with the node_id
    and timestamp before writing to the run record schema.
    """

    primitive: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome returned to the caller of a primitive."""

    primitive: str
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)


EventSink = Callable[[RecoveryEvent], None]


class AbortedRunError(RuntimeError):
    """Raised when ``abort_safely`` is invoked after the run has terminated."""


__all__ = [
    "AbortedRunError",
    "Driver",
    "EventSink",
    "GripperState",
    "Perception",
    "Pose",
    "RecoveryEvent",
    "RecoveryResult",
    "Safety",
]
