from __future__ import annotations

import json
import random
from pathlib import Path

from farm_edge_agent.drivers.base import GripperState, Pose

HOME_POSE: Pose = (300.0, 0.0, 300.0, 0.0, 0.0, 0.0)
HOME_JOINTS: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_VELOCITY_CAP = 100.0
SUB_STEPS = 10
LOG_PRECISION = 6


class LerobotMockDriver:
    """Sim arm for reviewers without hardware.

    Linear interpolation in Cartesian space, joint state from a fixed analytical
    map, gripper and e-stop tracked in memory. Every commanded action is rounded
    to ``LOG_PRECISION`` decimals and appended to ``trajectory_log_path`` as one
    JSON object per line so fixture diffs are byte-stable.
    """

    def __init__(
        self,
        seed: int = 0,
        trajectory_log_path: Path | None = None,
    ) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._trajectory_log_path = trajectory_log_path
        self._connected = False
        self._estop_armed = True
        self._tcp_pose: Pose = HOME_POSE
        self._joints: list[float] = list(HOME_JOINTS)
        self._gripper: GripperState = "open"
        if self._trajectory_log_path is not None:
            self._trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._trajectory_log_path.write_text("")

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def move_to(self, pose: Pose, velocity_cap: float = DEFAULT_VELOCITY_CAP) -> None:
        start = self._tcp_pose
        for step in range(1, SUB_STEPS + 1):
            t = step / SUB_STEPS
            interp: Pose = tuple(  # type: ignore[assignment]
                round(start[i] + t * (pose[i] - start[i]), LOG_PRECISION)
                for i in range(6)
            )
            self._tcp_pose = interp
            self._joints = _pseudo_ik(interp)
            self._log_action(
                {
                    "type": "move_to",
                    "step": step,
                    "pose": list(interp),
                    "joints": [round(j, LOG_PRECISION) for j in self._joints],
                    "velocity_cap": round(float(velocity_cap), LOG_PRECISION),
                }
            )

    def read_joint_state(self) -> list[float]:
        return list(self._joints)

    def read_tcp_pose(self) -> Pose:
        return self._tcp_pose

    def set_gripper(self, state: GripperState) -> None:
        self._gripper = state
        self._log_action({"type": "set_gripper", "state": state})

    @property
    def gripper_state(self) -> GripperState:
        return self._gripper

    def is_estop_armed(self) -> bool:
        return self._estop_armed

    def home(self) -> None:
        self.move_to(HOME_POSE, velocity_cap=DEFAULT_VELOCITY_CAP)
        self.set_gripper("open")

    def _log_action(self, action: dict) -> None:
        if self._trajectory_log_path is None:
            return
        line = json.dumps(action, sort_keys=True, separators=(",", ":"))
        with self._trajectory_log_path.open("a") as f:
            f.write(line + "\n")


def _pseudo_ik(pose: Pose) -> list[float]:
    """Toy analytical pose-to-joints map.

    Not real kinematics — a fixed function chosen so the same TCP pose always
    yields the same joint vector. Sufficient for a mock where downstream code
    needs stable read_joint_state() output.
    """
    x, y, z, rx, ry, rz = pose
    return [
        round(x * 0.001, LOG_PRECISION),
        round(y * 0.001, LOG_PRECISION),
        round(z * 0.001, LOG_PRECISION),
        round(rx, LOG_PRECISION),
        round(ry, LOG_PRECISION),
        round(rz, LOG_PRECISION),
    ]
