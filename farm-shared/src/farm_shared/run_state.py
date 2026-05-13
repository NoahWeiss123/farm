from dataclasses import dataclass
from typing import Literal

GripperState = Literal["open", "closed", "grasping"]


@dataclass
class RunState:
    """Handoff payload passed to the next backend at a fallback boundary.

    Mirrors the shape described in DESIGN.md "State handoff at fallback boundaries":
    the new backend must be able to plan its first chunk as if starting from this
    recovered state, so we ship pose, gripper, progress, and a fresh observation.
    """

    joint_pose: list[float]
    tcp_pose: tuple[list[float], list[float]]
    gripper_state: GripperState
    task_progress: int
    last_completed_chunk: int
    observation_snapshot: dict[str, str]
    critic_summary: str | None
