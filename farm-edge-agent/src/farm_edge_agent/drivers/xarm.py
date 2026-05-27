"""UFactory xArm driver for the Edge Agent.

Wraps ``xarm.wrapper.XArmAPI`` (via :mod:`_xarm_sdk_shim`) behind the
:class:`farm_edge_agent.drivers.base.Driver` protocol. Pose values are in
millimetres and degrees, base frame. ``move_to`` takes a pose *delta*, applied
via ``set_position(..., relative=True)``. The envelope check runs on the
computed absolute target before any SDK call so a bad delta never reaches the
arm.
"""

from __future__ import annotations

from time import monotonic
from typing import Any

from farm_edge_agent.drivers import _xarm_sdk_shim
from farm_edge_agent.drivers.base import GripperState, Pose

DEFAULT_HOME_POSE_850: Pose = (200.0, 0.0, 300.0, 180.0, 0.0, 0.0)
"""Forward, ~30cm above table, TCP pointing down. Inside the default envelope."""

DEFAULT_ENVELOPE_MIN: tuple[float, float, float] = (100.0, -200.0, 50.0)
DEFAULT_ENVELOPE_MAX: tuple[float, float, float] = (500.0, 200.0, 450.0)
"""Conservative 40cm cube centred ~30cm in front of the base. DESIGN → Safety."""

_DEFAULT_VELOCITY_CAP = 100.0

_SDK_OK = 0
_SDK_TIMEOUT_CODE = 9

_STATE_SPORT = 1
_STATE_PAUSE = 2

_GRIPPER_POSITIONS: dict[GripperState, int] = {
    "open": 850,
    "closed": 0,
    "grasping": 400,
}


class XArmDriverError(Exception):
    """FarmError-shaped error raised by the xArm driver.

    Mirrors ``farm_edge_agent.errors.FarmError``'s ``[FARM-Exxxx] ...`` surface
    so callers up the stack can treat driver failures the same as other
    catalogued errors. String codes are placeholders until dedicated slots
    land in the shared catalog; see ``tasks/_followups.md``.
    """

    def __init__(self, code: str, message: str, fix: str | None = None) -> None:
        self.code = code
        self.message = message
        self.fix = fix
        body = f"[{code}] {message}"
        if fix:
            body += f" — fix: {fix}"
        super().__init__(body)


class XArmDriver:
    """xArm SDK driver. Conforms to :class:`Driver`."""

    def __init__(
        self,
        arm_ip: str,
        *,
        home_pose: Pose = DEFAULT_HOME_POSE_850,
        envelope_min: tuple[float, float, float] = DEFAULT_ENVELOPE_MIN,
        envelope_max: tuple[float, float, float] = DEFAULT_ENVELOPE_MAX,
    ) -> None:
        self._ip = arm_ip
        self._home_pose = home_pose
        self._envelope_min = envelope_min
        self._envelope_max = envelope_max
        self._api: Any = None

    def connect(self, *, timeout: float = 30.0) -> None:
        api = _xarm_sdk_shim.XArmAPI(self._ip)
        for op, args, kwargs in (
            ("motion_enable", (), {"enable": True}),
            ("set_mode", (0,), {}),
            ("set_state", (0,), {}),
            # Without these the gripper SDK calls silently no-op:
            # set_gripper_position appears to succeed but no motion
            # occurs, and get_gripper_position returns whatever the jaws
            # happen to be parked at — which on the dashboard reads as a
            # stuck "Gripper bar = 100%" if the gripper happens to be
            # closed at boot.
            ("set_gripper_mode", (0,), {}),
            ("set_gripper_enable", (True,), {}),
            ("set_gripper_speed", (2000,), {}),
        ):
            code = self._call(op, timeout, getattr(api, op), *args, **kwargs)
            self._check_motion_code(op, code, timeout)
        self._api = api

    def disconnect(self, *, timeout: float = 5.0) -> None:
        if self._api is None:
            return
        self._call("disconnect", timeout, self._api.disconnect)
        self._api = None

    def move_to(
        self,
        pose: Pose,
        velocity_cap: float = _DEFAULT_VELOCITY_CAP,
        *,
        timeout: float = 10.0,
        wait: bool = True,
    ) -> None:
        api = self._require_api()
        current = self._read_pose(api, timeout=timeout)
        target: Pose = (
            current[0] + pose[0],
            current[1] + pose[1],
            current[2] + pose[2],
            current[3] + pose[3],
            current[4] + pose[4],
            current[5] + pose[5],
        )
        self._check_envelope(target)
        code = self._call(
            "set_position",
            timeout,
            api.set_position,
            *pose,
            relative=True,
            speed=velocity_cap,
            wait=wait,
            timeout=timeout,
        )
        self._check_motion_code("set_position", code, timeout)

    def read_joint_state(self, *, timeout: float = 5.0) -> list[float]:
        api = self._require_api()
        code, joints = self._call(
            "get_servo_angle", timeout, api.get_servo_angle, is_radian=False
        )
        if code != _SDK_OK:
            raise XArmDriverError(
                "FARM-E1009", f"xArm get_servo_angle returned code {code}"
            )
        return list(joints)

    def read_tcp_pose(self, *, timeout: float = 5.0) -> Pose:
        api = self._require_api()
        return self._read_pose(api, timeout=timeout)

    def set_gripper(
        self,
        state: GripperState,
        *,
        timeout: float = 5.0,
        wait: bool = True,
    ) -> None:
        api = self._require_api()
        position = _GRIPPER_POSITIONS[state]
        code = self._call(
            "set_gripper_position",
            timeout,
            api.set_gripper_position,
            position,
            wait=wait,
            timeout=timeout,
        )
        self._check_motion_code("set_gripper_position", code, timeout)

    def read_gripper_position(self, *, timeout: float = 2.0) -> float:
        """Raw gripper opening in the SDK's native scale (0 = closed,
        850 = fully open). Returns NaN if the SDK call fails so the
        caller can fall through to the commanded state."""
        api = self._require_api()
        try:
            code, pos = self._call(
                "get_gripper_position", timeout, api.get_gripper_position
            )
        except XArmDriverError:
            return float("nan")
        if code != _SDK_OK or pos is None:
            return float("nan")
        return float(pos)

    def is_estop_armed(self, *, timeout: float = 2.0) -> bool:
        api = self._require_api()
        code, state = self._call("get_state", timeout, api.get_state)
        if code != _SDK_OK:
            raise XArmDriverError(
                "FARM-E1009", f"xArm get_state returned code {code}"
            )
        return state in (_STATE_SPORT, _STATE_PAUSE)

    def home(self, *, timeout: float = 30.0, wait: bool = True) -> None:
        api = self._require_api()
        self._check_envelope(self._home_pose)
        code = self._call(
            "set_position",
            timeout,
            api.set_position,
            *self._home_pose,
            relative=False,
            wait=wait,
            timeout=timeout,
        )
        self._check_motion_code("home", code, timeout)

    def _require_api(self) -> Any:
        if self._api is None:
            raise XArmDriverError(
                "FARM-E1009",
                "xArm driver not connected",
                fix="call connect() before issuing commands",
            )
        return self._api

    def _read_pose(self, api: Any, *, timeout: float) -> Pose:
        code, pose = self._call(
            "get_position", timeout, api.get_position, is_radian=False
        )
        if code != _SDK_OK:
            raise XArmDriverError(
                "FARM-E1009", f"xArm get_position returned code {code}"
            )
        return (
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
        )

    def _check_envelope(self, target: Pose) -> None:
        x, y, z = target[0], target[1], target[2]
        in_envelope = (
            self._envelope_min[0] <= x <= self._envelope_max[0]
            and self._envelope_min[1] <= y <= self._envelope_max[1]
            and self._envelope_min[2] <= z <= self._envelope_max[2]
        )
        if not in_envelope:
            raise XArmDriverError(
                "FARM-E3001",
                (
                    f"Safety envelope violation: target ({x:.1f}, {y:.1f}, {z:.1f}) "
                    "mm outside workspace. Soft-stopped."
                ),
            )

    def _check_motion_code(self, op: str, code: int, timeout: float) -> None:
        if code == _SDK_OK:
            return
        if code == _SDK_TIMEOUT_CODE:
            raise XArmDriverError(
                "FARM-E1005",
                f"xArm '{op}' timed out after {timeout}s",
                fix="check arm power and network; restart the run",
            )
        raise XArmDriverError(
            "FARM-E1009", f"xArm '{op}' returned code {code}"
        )

    def _call(
        self,
        op: str,
        budget: float,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        start = monotonic()
        try:
            result = fn(*args, **kwargs)
        except XArmDriverError:
            raise
        except Exception as e:
            raise XArmDriverError(
                "FARM-E1009",
                f"xArm SDK '{op}' raised {type(e).__name__}: {e}",
            ) from e
        elapsed = monotonic() - start
        if elapsed > budget:
            raise XArmDriverError(
                "FARM-E1005",
                f"xArm SDK '{op}' exceeded {budget}s budget (took {elapsed:.2f}s)",
                fix="check arm power and network; restart the run",
            )
        return result
