"""Backend implementations for the FARM Edge Agent.

The ``Backend`` / ``Controller`` / ``Planner`` protocols in :mod:`.base` mirror
the role split from DESIGN.md "Backend types and capability cards".
:class:`~farm_edge_agent.backends.classical.ClassicalController` is the only
backend that runs locally in the Edge Agent; learned controllers (π0.5,
Gemini Robotics) live behind the Dispatcher in the cloud.
"""

from farm_edge_agent.backends.base import (
    ActionChunk,
    Backend,
    Controller,
    GripperState,
    Observation,
    PlanDAG,
    Planner,
    Pose,
    Waypoint,
)
from farm_edge_agent.backends.classical import (
    ClassicalController,
    ClassicalSkillNotFoundError,
    PerceptionCallable,
)

__all__ = [
    "ActionChunk",
    "Backend",
    "ClassicalController",
    "ClassicalSkillNotFoundError",
    "Controller",
    "GripperState",
    "Observation",
    "PerceptionCallable",
    "PlanDAG",
    "Planner",
    "Pose",
    "Waypoint",
]
