"""Lean MuJoCo backend for the UF850.

Loads ``assets/urdf/uf850/uf850.mjcf``, steps physics, reports joint state +
TCP pose, renders the three on-scene cameras (exterior, wrist, topdown), and
exposes a cartesian jog primitive that the dashboard buttons and (later) the
Quest teleop bridge both call into.

Pose convention crossing the public API is millimetres + radians, matching the
``Driver`` protocol; everything internal stays SI (metres, radians). This is
a fresh rewrite — no soft-grasp carry, no prop/scene system, no openpi
observation shaping. Add those back behind feature flags when they're needed
again.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from farm_edge_agent.drivers.base import GripperState, Pose

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "urdf" / "uf850"
MJCF_PATH = ASSETS_DIR / "uf850.mjcf"

HOME_JOINTS: tuple[float, ...] = (0.0, -0.50, -0.50, 0.0, -math.pi / 2, 0.0)
HOME_POSE: Pose = (0.0, -768.0, 270.0, math.pi, 0.0, 0.0)
DEFAULT_VELOCITY_CAP = 100.0

_ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
_ARM_ACTUATORS = tuple(f"a_{n}" for n in _ARM_JOINTS)
_FINGER_JOINT = "left_finger_joint"
_GRIPPER_ACTUATOR = "a_gripper"
_GRIPPER_OPEN_CTRL = 0.0
_GRIPPER_CLOSED_CTRL = 0.024
_TCP_SITE = "tcp"

JogAxis = Literal["x", "y", "z", "rx", "ry", "rz"]
EventCallback = Callable[[dict], None]


@dataclass(frozen=True)
class _IKResult:
    qpos: np.ndarray
    converged: bool
    pos_err: float
    rot_err: float
    iterations: int


class Sim:
    """Minimal UF850 MuJoCo backend.

    Implements the ``Driver`` protocol so a real-arm driver can swap in via
    a one-line change in the supervisor. The supervisor and HTTP/ROS surfaces
    only ever talk to this class.
    """

    def __init__(
        self,
        mjcf_path: Path = MJCF_PATH,
        event_sink: EventCallback | None = None,
        sim_rate: int = 25,
        render_height: int = 480,
        render_width: int = 640,
        realtime: bool = True,
        realtime_speed: float = 1.0,
    ) -> None:
        self._mjcf_path = Path(mjcf_path)
        self._event_sink = event_sink
        self._sim_rate = max(1, int(sim_rate))
        self._render_height = int(render_height)
        self._render_width = int(render_width)
        self._realtime = bool(realtime)
        self._realtime_speed = max(0.01, float(realtime_speed))

        self._lock = threading.RLock()
        self._connected = False
        self._estop_armed = True
        self._gripper: GripperState = "open"

        self._model = mujoco.MjModel.from_xml_string(
            self._mjcf_path.read_text(),
            _load_mesh_assets(self._mjcf_path),
        )
        self._data = mujoco.MjData(self._model)

        self._arm_jids = [_jid(self._model, n) for n in _ARM_JOINTS]
        self._arm_qadr = [int(self._model.jnt_qposadr[j]) for j in self._arm_jids]
        self._arm_dofadr = [int(self._model.jnt_dofadr[j]) for j in self._arm_jids]
        self._arm_aids = [_aid(self._model, n) for n in _ARM_ACTUATORS]
        self._gripper_aid = _aid(self._model, _GRIPPER_ACTUATOR)
        self._finger_qadr = int(
            self._model.jnt_qposadr[_jid(self._model, _FINGER_JOINT)]
        )
        self._tcp_sid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, _TCP_SITE
        )
        if self._tcp_sid < 0:
            raise RuntimeError(f"MJCF missing required site '{_TCP_SITE}'")

        self._apply_arm_qpos(np.asarray(HOME_JOINTS, dtype=np.float64))
        self._data.ctrl[self._gripper_aid] = _GRIPPER_OPEN_CTRL

        # Renderers are lazy because constructing one allocates a GL context.
        self._renderers: dict[tuple[str, int, int], mujoco.Renderer] = {}

    # ── connection lifecycle ─────────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True
        self._emit("connected", {})

    def disconnect(self) -> None:
        self._connected = False
        with self._lock:
            for r in self._renderers.values():
                r.close()
            self._renderers.clear()
        self._emit("disconnected", {})

    # ── motion ──────────────────────────────────────────────────────────────

    def move_to(self, pose: Pose, velocity_cap: float = DEFAULT_VELOCITY_CAP) -> None:
        """Move the TCP to ``pose`` (mm + radians).

        Arm motion is *kinematic* — IK solves for joint targets, then we
        write them straight into ``data.qpos`` and ``mj_forward``. We don't
        spin the PD path tracker because the UF850's stock position-actuator
        gains aren't stiff enough to hold the arm against gravity, so each
        physics step compounds a small sag. For a jog-driven teleop UI we
        want the cursor to land exactly where it was pointed. The gripper
        still runs through real physics in ``set_gripper`` so future grasp
        work isn't blocked.
        """
        target_pos = np.array(pose[:3], dtype=np.float64) / 1000.0
        target_quat = _rpy_to_quat(*pose[3:])

        with self._lock:
            scratch = mujoco.MjData(self._model)
            scratch.qpos[:] = self._data.qpos
            mujoco.mj_forward(self._model, scratch)
            result = self._ik(scratch, target_pos, target_quat)
            self._apply_arm_qpos(result.qpos)
        self._emit(
            "move_to",
            {
                "target_pose": list(pose),
                "ik_pos_err_m": result.pos_err,
                "ik_rot_err_rad": result.rot_err,
                "ik_iterations": result.iterations,
                "ik_converged": result.converged,
            },
        )
        _ = velocity_cap  # accepted for API compat

    def jog(
        self,
        axis: JogAxis,
        sign: int,
        *,
        step_mm: float = 20.0,
        step_rad: float = math.radians(10.0),
        velocity_cap: float = DEFAULT_VELOCITY_CAP,
    ) -> Pose:
        """Step the TCP by one jog increment along ``axis``.

        Translations move in the arm's base frame; rotations apply to the
        TCP's roll/pitch/yaw. Returns the new TCP pose.
        """
        if axis not in ("x", "y", "z", "rx", "ry", "rz"):
            raise ValueError(f"unknown jog axis: {axis!r}")
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        cur = self.read_tcp_pose()
        x, y, z, rx, ry, rz = cur
        if axis == "x":
            x += sign * step_mm
        elif axis == "y":
            y += sign * step_mm
        elif axis == "z":
            z += sign * step_mm
        elif axis == "rx":
            rx += sign * step_rad
        elif axis == "ry":
            ry += sign * step_rad
        elif axis == "rz":
            rz += sign * step_rad
        new_pose: Pose = (x, y, z, rx, ry, rz)
        self.move_to(new_pose, velocity_cap=velocity_cap)
        self._emit("jog", {"axis": axis, "sign": sign, "pose": list(new_pose)})
        return self.read_tcp_pose()

    def move_joint(
        self, qpos: list[float] | tuple[float, ...], velocity_cap: float = DEFAULT_VELOCITY_CAP
    ) -> None:
        with self._lock:
            self._apply_arm_qpos(np.asarray(qpos, dtype=np.float64))
        self._emit("move_joint", {"target_qpos": list(qpos)})
        _ = velocity_cap

    def home(self) -> None:
        with self._lock:
            self._apply_arm_qpos(np.asarray(HOME_JOINTS, dtype=np.float64))
            self._data.ctrl[self._gripper_aid] = _GRIPPER_OPEN_CTRL
            self._gripper = "open"
        self._emit("home", {})

    def set_gripper(self, state: GripperState) -> None:
        with self._lock:
            self._data.ctrl[self._gripper_aid] = (
                _GRIPPER_OPEN_CTRL if state == "open" else _GRIPPER_CLOSED_CTRL
            )
            self._step_gripper(self._sim_rate * 2)
            self._gripper = state
        self._emit("set_gripper", {"state": state})

    # ── observation ─────────────────────────────────────────────────────────

    def read_joint_state(self) -> list[float]:
        with self._lock:
            return [float(self._data.qpos[i]) for i in self._arm_qadr]

    def read_tcp_pose(self) -> Pose:
        with self._lock:
            mujoco.mj_forward(self._model, self._data)
            pos = self._data.site_xpos[self._tcp_sid].copy()
            mat = self._data.site_xmat[self._tcp_sid].reshape(3, 3).copy()
        rx, ry, rz = _mat_to_rpy(mat)
        return (
            float(pos[0]) * 1000.0,
            float(pos[1]) * 1000.0,
            float(pos[2]) * 1000.0,
            float(rx),
            float(ry),
            float(rz),
        )

    def read_gripper(self) -> float:
        """Gripper opening in [0=open, 1=closed]."""
        with self._lock:
            f = float(self._data.qpos[self._finger_qadr])
        return max(0.0, min(1.0, f / 0.035))

    @property
    def gripper_state(self) -> GripperState:
        return self._gripper

    def is_estop_armed(self) -> bool:
        return self._estop_armed

    def snapshot(self) -> dict:
        with self._lock:
            mujoco.mj_forward(self._model, self._data)
            joints = [float(self._data.qpos[i]) for i in self._arm_qadr]
            pos = self._data.site_xpos[self._tcp_sid].copy()
            mat = self._data.site_xmat[self._tcp_sid].reshape(3, 3).copy()
            grip01 = float(self._data.qpos[self._finger_qadr]) / 0.035
            t = float(self._data.time)
        rx, ry, rz = _mat_to_rpy(mat)
        return {
            "joints": joints,
            "tcp_pos_mm": [float(pos[0]) * 1000.0, float(pos[1]) * 1000.0, float(pos[2]) * 1000.0],
            "tcp_rpy": [float(rx), float(ry), float(rz)],
            "gripper": self._gripper,
            "gripper_pos": max(0.0, min(1.0, grip01)),
            "t": t,
        }

    def render_rgb(
        self,
        camera: str = "exterior",
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        h = height or self._render_height
        w = width or self._render_width
        key = (camera, h, w)
        with self._lock:
            renderer = self._renderers.get(key)
            if renderer is None:
                renderer = mujoco.Renderer(self._model, height=h, width=w)
                self._renderers[key] = renderer
            mujoco.mj_forward(self._model, self._data)
            renderer.update_scene(self._data, camera=camera)
            return renderer.render()

    # ── internals ──────────────────────────────────────────────────────────

    def _apply_arm_qpos(self, target: np.ndarray) -> None:
        """Write arm qpos directly and refresh derived state.

        Kinematic — no physics step on the arm. Joint limits are clipped
        with a small safety margin to keep the IK solver from parking the
        arm against a singularity it then can't escape on the next jog.
        Ctrl mirrors qpos so a future switch to PD-driven motion holds in
        place rather than snapping.

        Holds ``self._lock`` (reentrant) for the whole write+forward block
        so a concurrent ``render_rgb`` from another thread doesn't observe
        a half-written ``data.qpos`` while the GL renderer is mid-frame —
        the source of an earlier SIGTRAP on the shadow-sim render path.
        """
        target = np.asarray(target, dtype=np.float64)
        if target.shape[0] < 6:
            raise ValueError(f"qpos target must have >=6 dims; got {target.shape[0]}")
        target = target[:6].copy()
        with self._lock:
            for k, jid in enumerate(self._arm_jids):
                lo, hi = self._model.jnt_range[jid]
                target[k] = max(lo + 1e-3, min(hi - 1e-3, target[k]))
            for k, qadr in enumerate(self._arm_qadr):
                self._data.qpos[qadr] = float(target[k])
            # Zero arm velocity so any subsequent mj_step (e.g. during gripper
            # close) starts from rest rather than carrying stale momentum.
            for dofadr in self._arm_dofadr:
                self._data.qvel[dofadr] = 0.0
            for k, aid in enumerate(self._arm_aids):
                self._data.ctrl[aid] = float(target[k])
            mujoco.mj_forward(self._model, self._data)
        self._emit_joint_state()

    def _step_gripper(self, n: int) -> None:
        """Step physics ``n`` times — used by ``set_gripper`` so the
        finger joints actually close on whatever's under them."""
        step_dt = float(self._model.opt.timestep) / self._realtime_speed
        for _ in range(n):
            t0 = time.perf_counter()
            mujoco.mj_step(self._model, self._data)
            if self._realtime:
                spent = time.perf_counter() - t0
                if spent < step_dt:
                    time.sleep(step_dt - spent)

    def _ik(
        self,
        scratch: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        max_iter: int = 150,
        pos_tol: float = 2e-3,
        rot_tol: float = 5e-2,
        damping: float = 2e-2,
        step_scale: float = 0.5,
    ) -> _IKResult:
        """Damped-least-squares IK against the TCP site.

        Operates in the 6-DOF arm subspace — only the columns of the
        Jacobian corresponding to the arm's dof addresses are kept. This
        leaves the finger joints free to be driven independently by the
        gripper actuator without IK fighting them.
        """
        n_arm = 6
        jac_pos = np.zeros((3, self._model.nv))
        jac_rot = np.zeros((3, self._model.nv))

        for it in range(max_iter):
            mujoco.mj_forward(self._model, scratch)
            cur_pos = scratch.site_xpos[self._tcp_sid].copy()
            cur_mat = scratch.site_xmat[self._tcp_sid].reshape(3, 3).copy()
            cur_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cur_quat, cur_mat.flatten())

            err_pos = target_pos - cur_pos
            err_rot = _quat_error(cur_quat, target_quat)
            pos_err_n = float(np.linalg.norm(err_pos))
            rot_err_n = float(np.linalg.norm(err_rot))
            if pos_err_n < pos_tol and rot_err_n < rot_tol:
                return _IKResult(
                    qpos=np.array([scratch.qpos[i] for i in self._arm_qadr]),
                    converged=True,
                    pos_err=pos_err_n,
                    rot_err=rot_err_n,
                    iterations=it,
                )

            mujoco.mj_jacSite(self._model, scratch, jac_pos, jac_rot, self._tcp_sid)
            J = np.vstack([jac_pos[:, self._arm_dofadr], jac_rot[:, self._arm_dofadr]])
            err = np.concatenate([err_pos, err_rot])
            # Damped pseudo-inverse: dq = J^T (J J^T + λ²I)^-1 e
            JJt = J @ J.T + (damping**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            dq *= step_scale
            for k, qadr in enumerate(self._arm_qadr):
                scratch.qpos[qadr] += float(dq[k])
            # Clamp to limits during the search
            for k, jid in enumerate(self._arm_jids):
                lo, hi = self._model.jnt_range[jid]
                scratch.qpos[self._arm_qadr[k]] = max(
                    lo + 1e-3, min(hi - 1e-3, float(scratch.qpos[self._arm_qadr[k]]))
                )
            _ = n_arm  # purely for readability above

        # Bail out without converging — return best-effort qpos. The caller
        # decides whether to honor it; the dashboard surfaces ik_converged.
        return _IKResult(
            qpos=np.array([scratch.qpos[i] for i in self._arm_qadr]),
            converged=False,
            pos_err=pos_err_n,
            rot_err=rot_err_n,
            iterations=max_iter,
        )

    def _emit_joint_state(self) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            {
                "type": "joint_state",
                "joints": [float(self._data.qpos[i]) for i in self._arm_qadr],
                "gripper_pos": max(0.0, min(1.0, float(self._data.qpos[self._finger_qadr]) / 0.035)),
                "t": float(self._data.time),
            }
        )

    def _emit(self, kind: str, payload: dict) -> None:
        if self._event_sink is None:
            return
        self._event_sink({"type": kind, **payload, "t": float(self._data.time)})


# ── helpers ─────────────────────────────────────────────────────────────────


def _jid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _aid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _load_mesh_assets(mjcf_path: Path) -> dict[str, bytes]:
    """Map mesh filenames to bytes so MuJoCo can compile from a string.

    The MJCF references meshes via ``meshdir=meshes/visual`` and via the
    sibling ``../gripper/`` path for the gripper STLs.
    """
    assets: dict[str, bytes] = {}
    base = mjcf_path.parent / "meshes"
    visual = base / "visual"
    if visual.is_dir():
        for f in visual.glob("*.stl"):
            assets[f.name] = f.read_bytes()
    gripper = base / "gripper"
    if gripper.is_dir():
        for f in gripper.glob("*.stl"):
            assets[f"../gripper/{f.name}"] = f.read_bytes()
    return assets


def _rpy_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ-Euler → quat (w, x, y, z) in MuJoCo convention."""
    cx, cy, cz = math.cos(rx / 2), math.cos(ry / 2), math.cos(rz / 2)
    sx, sy, sz = math.sin(rx / 2), math.sin(ry / 2), math.sin(rz / 2)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ]
    )


def _mat_to_rpy(mat: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    if sy < 1e-6:
        rx = math.atan2(-mat[1, 2], mat[1, 1])
        ry = math.atan2(-mat[2, 0], sy)
        rz = 0.0
    else:
        rx = math.atan2(mat[2, 1], mat[2, 2])
        ry = math.atan2(-mat[2, 0], sy)
        rz = math.atan2(mat[1, 0], mat[0, 0])
    return rx, ry, rz


def _quat_error(cur: np.ndarray, target: np.ndarray) -> np.ndarray:
    """3-vector axis-angle of (target * conj(cur)) — the rotation we need
    to apply to ``cur`` to land on ``target``. Used as the orientation
    error fed into the IK jacobian step."""
    cur_inv = np.array([cur[0], -cur[1], -cur[2], -cur[3]])
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, target, cur_inv)
    if dq[0] < 0:
        dq = -dq
    axis = dq[1:]
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.zeros(3)
    angle = 2.0 * math.atan2(n, dq[0])
    return axis * (angle / n)
