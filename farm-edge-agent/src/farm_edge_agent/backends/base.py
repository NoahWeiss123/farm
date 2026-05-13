"""Backend protocols and shared data types.

Mirrors the ``Planner`` / ``Controller`` protocols from DESIGN.md "Backend types
and capability cards". ``Backend`` is the shared marker that just requires a
capability card; the role-specific protocols extend it with the method the
Dispatcher actually calls.

``Observation`` and ``ActionChunk`` are the wire-shape types the Edge Agent's
control loop and the cloud Dispatcher both speak. They are deliberately small —
camera frames live in ``Observation.frames`` as bytes, joint state as a plain
tuple — so a chunk fits inside the run-record's 4 KiB event budget without a
sidecar reference.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from farm_shared.capability_card import CapabilityCard

GripperState = Literal["open", "closed", "grasping"]

Pose = tuple[float, float, float, float, float, float]
"""TCP pose in the arm's base frame: ``(x, y, z, rx, ry, rz)``. mm and degrees."""


@dataclass(frozen=True)
class Observation:
    """One observation tick from the Edge Agent's perception layer.

    ``scene`` is an optional perception-derived dict of named object poses; the
    classical controller reads it directly, learned controllers ignore it and
    look at ``frames`` instead. Either may be empty for a given tick.
    """

    joint_state: tuple[float, ...]
    tcp_pose: Pose
    gripper: GripperState
    frames: dict[str, bytes] = field(default_factory=dict)
    scene: dict[str, Pose] = field(default_factory=dict)


@dataclass(frozen=True)
class Waypoint:
    """A single Cartesian target inside an :class:`ActionChunk`.

    Pose is absolute base-frame TCP. The executor converts to the wire-protocol
    delta encoding at send time; keeping the controller in absolute mm keeps
    skill code readable and makes test diffs human-checkable.
    """

    pose: Pose
    gripper: GripperState | None = None
    label: str = ""


@dataclass(frozen=True)
class ActionChunk:
    """A planned slice of motion the controller hands to the Edge Agent.

    A skill emits multiple chunks; the executor runs them back-to-back from the
    chunk buffer at the control-rate cadence. ``terminal`` marks the final
    chunk for a skill so the Dispatcher knows the node is complete.
    """

    skill: str
    step_index: int
    waypoints: tuple[Waypoint, ...]
    terminal: bool = False


PlanDAG = dict[str, Any]
"""Placeholder for the router-emitted plan DAG.

The concrete schema lands with the Planner Worker (task 018); the protocol
below uses this alias so the seam exists in the right place today.
"""


@runtime_checkable
class Backend(Protocol):
    """Shared marker every FARM backend role implements."""

    capability_card: CapabilityCard


@runtime_checkable
class Controller(Backend, Protocol):
    """Emits action chunks given a stream of observations + a per-node instruction."""

    def act(
        self,
        obs_stream: AsyncIterator[Observation],
        instruction: str,
    ) -> AsyncIterator[ActionChunk]: ...


@runtime_checkable
class Planner(Backend, Protocol):
    """Produces a plan DAG from a task string + initial scene observation."""

    async def plan(self, task: str, scene: Observation) -> PlanDAG: ...
