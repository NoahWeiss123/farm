"""Backend protocol — the contract the supervisor talks to.

Two implementations today: ``SimBackend`` (MuJoCo) and ``XArmBackend``
(UF850 over the xArm Python SDK). Both expose snapshots in the same units
(mm + radians + 0/1 gripper) so the dashboard and ROS-TCP bridge don't
need to know which hardware is behind them.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

JogAxis = Literal["x", "y", "z", "rx", "ry", "rz"]
GripperState = Literal["open", "closed", "grasping"]


class RobotBackend(Protocol):
    """Uniform surface every backend implements.

    Snapshots use mm + radians; jog steps use mm + radians; gripper state
    is the same Literal as the underlying driver. Backends own their
    own connection lifecycle and any background polling threads.
    """

    backend_name: str

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def snapshot(self) -> dict[str, Any]:
        """Combined state: joints (rad), tcp_pos_mm, tcp_rpy (rad),
        gripper (Literal), gripper_pos (0=open, 1=closed), t (epoch s)."""
        ...

    def jog(
        self,
        axis: JogAxis,
        sign: int,
        *,
        step_mm: float,
        step_rad: float,
    ) -> dict[str, Any]:
        """Apply a single jog increment along ``axis``. Returns
        ``{"pose": [..6], "snapshot": {...}}`` after the move settles."""
        ...

    def home(self) -> dict[str, Any]:
        """Drive arm to a backend-defined home pose. Returns snapshot."""
        ...

    def set_gripper(self, state: GripperState) -> dict[str, Any]:
        """Open / close the gripper. Returns snapshot."""
        ...

    def estop(self) -> dict[str, Any]:
        """Software emergency stop. Halts motion, rejects further jog/home
        until ``estop_clear()`` is called. Returns ``{"estopped": bool}``."""
        ...

    def estop_clear(self) -> dict[str, Any]:
        """Re-arm the backend after an e-stop."""
        ...
