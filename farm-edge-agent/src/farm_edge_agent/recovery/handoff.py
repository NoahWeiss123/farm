"""RunState — the snapshot passed across a fallback boundary.

When the Dispatcher walks from one backend to the next, the new backend has
to know where the arm is and how much of the plan has actually completed.
``RunState`` is that snapshot. See DESIGN.md → Fallback chain → State handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import Driver, GripperState, Pose, Safety


@dataclass(frozen=True)
class RunState:
    """Frozen snapshot of the arm and plan at a handoff boundary."""

    run_id: str
    joint_state: list[float]
    tcp_pose: Pose
    gripper_state: GripperState
    plan: dict[str, Any]
    task_progress_index: int
    last_chunk_index: int
    home_pose: Pose
    velocity_cap: float
    observation: dict[str, Any] = field(default_factory=dict)
    critic_summary: str | None = None

    @classmethod
    def from_current(
        cls,
        driver: Driver,
        safety: Safety,
        run_id: str,
        plan: dict[str, Any],
        last_chunk: int,
        *,
        task_progress_index: int = 0,
        observation: dict[str, Any] | None = None,
        critic_summary: str | None = None,
    ) -> RunState:
        """Snapshot the live arm state for handoff to the next backend."""

        return cls(
            run_id=run_id,
            joint_state=list(driver.read_joint_state()),
            tcp_pose=driver.read_tcp_pose(),
            gripper_state=driver.read_gripper_state(),
            plan=dict(plan),
            task_progress_index=task_progress_index,
            last_chunk_index=last_chunk,
            home_pose=safety.home_pose,
            velocity_cap=safety.velocity_cap,
            observation=dict(observation) if observation is not None else {},
            critic_summary=critic_summary,
        )


__all__ = ["RunState"]
