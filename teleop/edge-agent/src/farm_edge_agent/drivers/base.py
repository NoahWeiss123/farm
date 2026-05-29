from __future__ import annotations

from typing import Literal, Protocol

GripperState = Literal["open", "closed", "grasping"]

Pose = tuple[float, float, float, float, float, float]
"""TCP pose in the arm's base frame: (x, y, z, rx, ry, rz). mm and radians."""


class Driver(Protocol):
    """Contract every arm driver implements.

    Concrete drivers (xarm, franka, lerobot-mock) plug into the Edge Agent
    behind this protocol. The Edge Agent never talks to a vendor SDK directly
    so swapping sim for hardware is a one-line config change.
    """

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def move_to(self, pose: Pose, velocity_cap: float) -> None: ...

    def read_joint_state(self) -> list[float]: ...

    def read_tcp_pose(self) -> Pose: ...

    def set_gripper(self, state: GripperState) -> None: ...

    def is_estop_armed(self) -> bool: ...

    def home(self) -> None: ...
