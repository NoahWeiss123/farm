"""MuJoCo-backed SimDriver for the UFactory 850.

Implements the Edge Agent ``Driver`` protocol against an in-process MuJoCo
simulation. Loads ``assets/urdf/uf850/uf850.mjcf``, advances physics on each
command, exposes joint state + TCP pose + RGB observations, and emits joint
state events for downstream consumers (RunRecord, UI live view).

Poses cross the protocol boundary in millimeters and radians to match the
existing Driver contract; internally everything is SI (meters, radians).
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from farm_edge_agent.drivers.base import GripperState, Pose
from farm_edge_agent.drivers.sim.ik import solve_ik

ASSETS_DIR = Path(__file__).resolve().parents[4] / "assets" / "urdf" / "uf850"
MJCF_PATH = ASSETS_DIR / "uf850.mjcf"

# Home: arm "ready", flange ~270 mm above the floor, gripper straight down.
# With the 205 mm robot-mount pedestal in the MJCF, this puts the TCP at
# roughly (0, -770, 270) mm — well above the workspace but inside the
# kinematic basin where the IK can reach floor-level grasps reliably.
HOME_JOINTS: tuple[float, ...] = (0.0, -0.50, -0.50, 0.0, -math.pi / 2, 0.0)
HOME_POSE: Pose = (0.0, -768.0, 270.0, math.pi, 0.0, 0.0)
DEFAULT_VELOCITY_CAP = 100.0
# Quaternion for "gripper +Z = world -Z" (180° rotation around world X).
GRIPPER_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)

_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# Simplified parallel-jaw: one slider (left_finger_joint); right finger
# mimics via an equality constraint. ctrl is the half-stroke in metres.
_FINGER_JOINTS = ("left_finger_joint",)
_GRIPPER_CTRL_OPEN = 0.0      # fingers retract — 70 mm jaw gap
_GRIPPER_CTRL_CLOSED = 0.024  # ~22 mm gap → grips a 25 mm block (1.5 mm compression)
_TCP_SITE = "tcp"


@dataclass
class Prop:
    """A box or cylinder prop placed in the workspace."""

    id: str
    shape: str  # "box" or "cylinder"
    size: tuple[float, ...]  # box: (sx, sy, sz); cylinder: (radius, half_height)
    pos: tuple[float, float, float]  # SI (meters)
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.05
    friction: tuple[float, float, float] = (1.5, 0.05, 0.001)


@dataclass
class Scene:
    name: str
    props: list[Prop] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> Scene:
        raw = json.loads(Path(path).read_text())
        props = [
            Prop(
                id=p["id"],
                shape=p["shape"],
                size=tuple(p["size"]),
                pos=tuple(p["pos"]),
                rgba=tuple(p.get("rgba", [0.8, 0.8, 0.8, 1.0])),
                quat=tuple(p.get("quat", [1.0, 0.0, 0.0, 0.0])),
                mass=float(p.get("mass", 0.05)),
                friction=tuple(p.get("friction", [1.5, 0.05, 0.001])),
            )
            for p in raw.get("props", [])
        ]
        return cls(name=raw.get("name", path.stem), props=props)


def load_scene(path: str | Path) -> Scene:
    return Scene.from_json(Path(path))


def _prop_to_xml(prop: Prop) -> str:
    """Render a Prop as a free-floating body with collision groups that let
    it interact with the floor, the arm/gripper, and other props.

    contype=2, conaffinity=7 (= floor(1) | prop(2) | arm(4))
    — see the collision-group block at the top of uf850.mjcf.
    """
    rgba = " ".join(f"{c:.3f}" for c in prop.rgba)
    fric = " ".join(f"{c:.4f}" for c in prop.friction)
    common = (
        f'mass="{prop.mass:.4f}" rgba="{rgba}" friction="{fric}" '
        f'condim="6" contype="2" conaffinity="7"'
    )
    if prop.shape == "box":
        sx, sy, sz = prop.size
        size = f"{sx:.4f} {sy:.4f} {sz:.4f}"
        geom = f'<geom type="box" size="{size}" {common}/>'
    elif prop.shape == "cylinder":
        radius, half_h = prop.size
        size = f"{radius:.4f} {half_h:.4f}"
        geom = f'<geom type="cylinder" size="{size}" {common}/>'
    else:
        raise ValueError(f"unsupported prop shape: {prop.shape}")
    pos = " ".join(f"{c:.4f}" for c in prop.pos)
    quat = " ".join(f"{c:.4f}" for c in prop.quat)
    return f'<body name="prop_{prop.id}" pos="{pos}" quat="{quat}"><freejoint/>{geom}</body>'


def _inject_props(mjcf_text: str, props: list[Prop]) -> str:
    if not props:
        return mjcf_text
    blob = "\n".join("    " + _prop_to_xml(p) for p in props)
    return mjcf_text.replace("</worldbody>", blob + "\n  </worldbody>")


EventCallback = Callable[[dict], None]


class SimDriver:
    """MuJoCo-backed driver. Drop-in for ``LerobotMockDriver`` with physics.

    Parameters
    ----------
    scene
        Optional Scene with props to inject into the world before compilation.
    mjcf_path
        Path to the MJCF file; defaults to the bundled UF850.
    event_sink
        Callable invoked with dict events ({"type": "joint_state", ...}, etc.)
        whenever the driver updates state. Used by the RunLoop to forward
        events to the run record and the live UI stream.
    sim_rate
        Substeps per command. With the MJCF's 2 ms timestep, 25 substeps
        gives a 50 ms control period (≈20 Hz).
    """

    def __init__(
        self,
        scene: Scene | None = None,
        mjcf_path: Path = MJCF_PATH,
        event_sink: EventCallback | None = None,
        sim_rate: int = 25,
        max_settle_steps: int = 400,
        render_height: int = 480,
        render_width: int = 640,
        camera: str | int = -1,
        grasp_radius_m: float = 0.04,
        realtime: bool = False,
        realtime_speed: float = 1.0,
    ) -> None:
        self._mjcf_path = Path(mjcf_path)
        self._scene = scene or Scene(name="empty")
        self._event_sink = event_sink
        self._sim_rate = max(1, int(sim_rate))
        self._max_settle_steps = int(max_settle_steps)
        self._render_height = int(render_height)
        self._render_width = int(render_width)
        self._camera = camera
        self._lock = threading.Lock()
        self._connected = False
        self._estop_armed = True
        self._gripper: GripperState = "open"
        # Soft-grasp state: name of prop currently attached to the gripper,
        # plus the relative offset (TCP frame → prop frame) at grasp time.
        self._grasp_radius_m = float(grasp_radius_m)
        self._grasped_prop_id: str | None = None
        self._grasp_offset_pos: np.ndarray | None = None
        self._grasp_offset_quat: np.ndarray | None = None
        self._realtime = bool(realtime)
        self._realtime_speed = max(0.01, float(realtime_speed))

        # Compile model with props injected at construction time
        base_xml = self._mjcf_path.read_text()
        full_xml = _inject_props(base_xml, self._scene.props)
        self._xml = full_xml
        # Use a temp file so MuJoCo can resolve relative meshdir
        self._model = mujoco.MjModel.from_xml_string(full_xml, _make_assets_dict(self._mjcf_path))
        self._data = mujoco.MjData(self._model)

        # Cache joint/actuator/site IDs
        self._arm_joint_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in _ARM_JOINT_NAMES
        ]
        self._arm_qpos_addrs = [
            int(self._model.jnt_qposadr[j]) for j in self._arm_joint_ids
        ]
        self._finger_joint_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in _FINGER_JOINTS
        ]
        self._finger_qpos_addrs = [
            int(self._model.jnt_qposadr[j]) for j in self._finger_joint_ids
        ]
        self._actuator_ids = {
            "arm": [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{n}")
                for n in _ARM_JOINT_NAMES
            ],
            "gripper": mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_gripper"
            ),
        }
        self._tcp_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, _TCP_SITE
        )
        if self._tcp_site_id < 0:
            raise RuntimeError(f"site {_TCP_SITE!r} not found in MJCF")

        # Initialize to home pose
        self._set_arm_joints(HOME_JOINTS, settle=False)
        self._set_gripper_targets(open_=True)
        mujoco.mj_forward(self._model, self._data)

        self._renderer: mujoco.Renderer | None = None

    # ── connection lifecycle ─────────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True
        self._emit("connected", {"scene": self._scene.name})

    def disconnect(self) -> None:
        self._connected = False
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._emit("disconnected", {})

    # ── motion primitives ────────────────────────────────────────────────────

    def move_to(
        self,
        pose: Pose,
        velocity_cap: float = DEFAULT_VELOCITY_CAP,
        path_steps: int = 16,
    ) -> None:
        """Move TCP to ``pose`` (mm + radians).

        Strategy:
          1. Solve a single IK against a **scratch copy** of MjData with the
             *current* qpos as warmstart and ``seed_attempts=1``. This
             forces the IK to find a target config near the current basin —
             no flipping to a far-away config that requires the arm to
             invert itself.
          2. Joint-space-interpolate from the live qpos to the IK target
             across ``path_steps`` substeps, with adequate settling.
          3. Hold the final ctrl so the gripper can act without arm drift.

        Full-quaternion orientation constraint — keeping joint5 inside its
        comfort zone is up to the home pose + the workspace bounds.
        """
        target_pos_m = np.array([pose[0], pose[1], pose[2]], dtype=np.float64) / 1000.0
        target_quat = _rpy_to_quat(pose[3], pose[4], pose[5])
        with self._lock:
            # Solve IK on a scratch MjData so the live state isn't perturbed.
            scratch = mujoco.MjData(self._model)
            scratch.qpos[:] = self._data.qpos
            scratch.qvel[:] = 0
            mujoco.mj_forward(self._model, scratch)
            result = solve_ik(
                self._model,
                scratch,
                self._tcp_site_id,
                target_pos_m,
                target_quat=target_quat,
                arm_joint_ids=self._arm_joint_ids,
                max_iter=200,
                pos_tol=2e-3,
                rot_tol=5e-2,
                seed_attempts=1,        # stay in the current basin
                step_scale=0.5,
                damping=2e-2,
            )
            target_qpos = result.qpos.tolist()
            current_qpos = [float(self._data.qpos[i]) for i in self._arm_qpos_addrs]
            # Joint-space cosine ease-in/out gives a smoother trajectory
            # than linear at the same step count — less PD chatter at the
            # acceleration/deceleration boundaries.
            for s in range(1, path_steps + 1):
                t = s / path_steps
                ease = 0.5 - 0.5 * math.cos(math.pi * t)
                interp = [
                    current_qpos[k] + (target_qpos[k] - current_qpos[k]) * ease
                    for k in range(6)
                ]
                # Real UF850 caps each joint at 180°/s; with torque-limited
                # actuators a 90° move takes ~0.5 s wall. settle_steps=200
                # × 2 ms = 0.4 s sim per substep is enough for the arm to
                # actually arrive.
                self._drive_arm_to(
                    interp,
                    velocity_cap=velocity_cap,
                    settle_steps=200,
                    hold_after=False,
                )
            # Extra settle at the final config so subsequent gripper
            # commands act on a stable arm.
            self._drive_arm_to(
                target_qpos,
                velocity_cap=velocity_cap,
                settle_steps=400,
                hold_after=True,
            )
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

    def move_joint(self, qpos: list[float], velocity_cap: float = DEFAULT_VELOCITY_CAP) -> None:
        """Direct joint-space command (radians)."""
        with self._lock:
            self._drive_arm_to(list(qpos), velocity_cap=velocity_cap)
        self._emit("move_joint", {"target_qpos": list(qpos)})

    def read_joint_state(self) -> list[float]:
        with self._lock:
            return [float(self._data.qpos[i]) for i in self._arm_qpos_addrs]

    def read_tcp_pose(self) -> Pose:
        with self._lock:
            mujoco.mj_forward(self._model, self._data)
            pos = self._data.site_xpos[self._tcp_site_id]
            mat = self._data.site_xmat[self._tcp_site_id].reshape(3, 3)
        rx, ry, rz = _mat_to_rpy(mat)
        return (
            float(pos[0]) * 1000.0,
            float(pos[1]) * 1000.0,
            float(pos[2]) * 1000.0,
            float(rx),
            float(ry),
            float(rz),
        )

    def set_gripper(self, state: GripperState) -> None:
        attached: str | None = None
        with self._lock:
            if state == "open":
                released = self._grasped_prop_id
                self._grasped_prop_id = None
                self._grasp_offset_pos = None
                self._grasp_offset_quat = None
                self._set_gripper_targets(open_=True)
                for _ in range(self._sim_rate * 2):
                    mujoco.mj_step(self._model, self._data)
                self._gripper = state
                if released is not None:
                    attached = released
                    self._emit("release_prop", {"prop": released})
            else:
                # Capture the grasp candidate *before* commanding the close,
                # because the close itself shifts the arm and props enough
                # that "closest prop" can change by the time the actuator
                # settles. The carry override then begins the next step.
                attached = self._try_grasp_closest_prop()
                self._set_gripper_targets(open_=False)
                for _ in range(self._sim_rate * 2):
                    mujoco.mj_step(self._model, self._data)
                    self._carry_grasped_prop()
                self._gripper = state
                if attached is not None:
                    self._emit("grasp_prop", {"prop": attached})
        self._emit("set_gripper", {"state": state, "attached": attached})

    @property
    def gripper_state(self) -> GripperState:
        return self._gripper

    def is_estop_armed(self) -> bool:
        return self._estop_armed

    def check_pose_reachable(self, pose: Pose) -> bool:
        """Cheap reachability probe — position-only IK on a scratch MjData.

        Accepts both the Driver protocol's millimeter Pose and the safety
        module's meter Pose: positions with |x|<5 are treated as meters,
        otherwise millimeters. Orientation is intentionally dropped — many
        UF850 poses clamp joint5 to its limit but ``move_to`` still
        succeeds, so the reachability gate should not reject them.
        """
        from farm_edge_agent.drivers.sim.ik import solve_ik

        raw = np.array([pose[0], pose[1], pose[2]], dtype=np.float64)
        target_pos_m = raw if float(np.max(np.abs(raw))) < 5.0 else raw / 1000.0
        scratch = mujoco.MjData(self._model)
        scratch.qpos[:] = self._data.qpos
        scratch.qvel[:] = 0
        scratch.ctrl[:] = self._data.ctrl
        mujoco.mj_forward(self._model, scratch)
        result = solve_ik(
            self._model,
            scratch,
            self._tcp_site_id,
            target_pos_m,
            target_quat=None,
            arm_joint_ids=self._arm_joint_ids,
            max_iter=120,
            pos_tol=5e-3,
            rot_tol=1.0,
            seed_attempts=3,
        )
        return result.pos_err < 0.03  # 30 mm slack

    def check_self_collision(self, pose: Pose) -> bool:
        """Sim path is permissive — MuJoCo handles contacts in physics.

        Real-arm drivers should implement this against vendor IK + collision
        primitives.
        """
        return False

    def home(self) -> None:
        with self._lock:
            self._drive_arm_to(list(HOME_JOINTS), velocity_cap=DEFAULT_VELOCITY_CAP)
            self._set_gripper_targets(open_=True)
            for _ in range(self._sim_rate):
                mujoco.mj_step(self._model, self._data)
            self._gripper = "open"
        self._emit("home", {})

    # ── observation ──────────────────────────────────────────────────────────

    def render_rgb(self, camera: str | int | None = None) -> np.ndarray:
        """Render an RGB observation. Lazy-creates a Renderer per driver."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, height=self._render_height, width=self._render_width
            )
        with self._lock:
            mujoco.mj_forward(self._model, self._data)
            self._renderer.update_scene(
                self._data, camera=camera if camera is not None else self._camera
            )
            img = self._renderer.render()
        return img

    def snapshot(self) -> dict:
        """Return a JSON-serializable observation: joints + tcp + prop poses."""
        with self._lock:
            mujoco.mj_forward(self._model, self._data)
            joints = [float(self._data.qpos[i]) for i in self._arm_qpos_addrs]
            tcp = self._data.site_xpos[self._tcp_site_id].tolist()
            tcp_mat = self._data.site_xmat[self._tcp_site_id].reshape(3, 3)
            tcp_quat = np.zeros(4)
            mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
            props = {}
            for prop in self._scene.props:
                body_name = f"prop_{prop.id}"
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    props[prop.id] = {
                        "pos": self._data.xpos[bid].tolist(),
                        "quat": self._data.xquat[bid].tolist(),
                    }
        return {
            "joints": joints,
            "tcp_pos_m": tcp,
            "tcp_quat": tcp_quat.tolist(),
            "gripper": self._gripper,
            "props": props,
        }

    # ── internals ────────────────────────────────────────────────────────────

    def _set_arm_joints(self, joints: list[float] | tuple[float, ...], settle: bool = True) -> None:
        for q_idx, val in zip(self._arm_qpos_addrs, joints, strict=False):
            self._data.qpos[q_idx] = float(val)
        for k, aid in enumerate(self._actuator_ids["arm"]):
            self._data.ctrl[aid] = float(joints[k])
        if settle:
            for _ in range(self._sim_rate):
                mujoco.mj_step(self._model, self._data)

    def _set_gripper_targets(self, open_: bool) -> None:
        # One actuator on the xArm Gripper drive_joint. 0 rad = fully open
        # (~85 mm jaw gap), 0.85 rad = fully closed. The 0.69 rad close
        # target gives a small clamping margin on a 25 mm block.
        aid = self._actuator_ids["gripper"]
        self._data.ctrl[aid] = (
            _GRIPPER_CTRL_OPEN if open_ else _GRIPPER_CTRL_CLOSED
        )

    def _drive_arm_to(
        self,
        qpos: list[float],
        velocity_cap: float,
        settle_steps: int | None = None,
        hold_after: bool = True,
    ) -> None:
        """Drive arm to target joints with PD, stepping the sim.

        ``hold_after`` controls what happens at the end of the settle:
          * True  → reset ctrl to the actual settled qpos. Use this at the
            end of a move so gripper actions don't fight an unreachable
            commanded target.
          * False → leave ctrl at the commanded ``qpos`` so a subsequent
            interpolation step continues from the right reference. Use
            this in joint-space path tracking.
        """
        for k, aid in enumerate(self._actuator_ids["arm"]):
            self._data.ctrl[aid] = float(qpos[k])
        budget = settle_steps if settle_steps is not None else self._max_settle_steps
        step_dt = float(self._model.opt.timestep) / self._realtime_speed
        for step in range(budget):
            t_step = time.perf_counter()
            mujoco.mj_step(self._model, self._data)
            self._carry_grasped_prop()
            if self._realtime:
                elapsed = time.perf_counter() - t_step
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)
            if step % self._sim_rate == 0:
                self._emit_joint_state()
            cur = np.array([self._data.qpos[i] for i in self._arm_qpos_addrs])
            err = float(np.linalg.norm(cur - np.array(qpos)))
            if err < 5e-4:
                break
        if hold_after:
            for k, aid in enumerate(self._actuator_ids["arm"]):
                self._data.ctrl[aid] = float(self._data.qpos[self._arm_qpos_addrs[k]])
        self._emit_joint_state()

    def _try_grasp_closest_prop(self) -> str | None:
        """If a prop is within grasp_radius of the TCP, attach it."""
        mujoco.mj_forward(self._model, self._data)
        tcp_pos = self._data.site_xpos[self._tcp_site_id].copy()
        tcp_mat = self._data.site_xmat[self._tcp_site_id].reshape(3, 3).copy()
        best: tuple[float, str, int] | None = None
        for prop in self._scene.props:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"prop_{prop.id}")
            if bid < 0:
                continue
            prop_pos = self._data.xpos[bid]
            d = float(np.linalg.norm(prop_pos - tcp_pos))
            if d < self._grasp_radius_m and (best is None or d < best[0]):
                best = (d, prop.id, bid)
        if best is None:
            return None
        _, prop_id, bid = best
        prop_pos = self._data.xpos[bid].copy()
        prop_quat = self._data.xquat[bid].copy()
        # Store offsets in TCP frame
        offset_pos_world = prop_pos - tcp_pos
        self._grasp_offset_pos = tcp_mat.T @ offset_pos_world
        # quat_offset = inv(tcp_quat) * prop_quat
        tcp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
        tcp_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(tcp_quat_inv, tcp_quat)
        offset_q = np.zeros(4)
        mujoco.mju_mulQuat(offset_q, tcp_quat_inv, prop_quat)
        self._grasp_offset_quat = offset_q
        self._grasped_prop_id = prop_id
        return prop_id

    def _carry_grasped_prop(self) -> None:
        if (
            self._grasped_prop_id is None
            or self._grasp_offset_pos is None
            or self._grasp_offset_quat is None
        ):
            return
        bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, f"prop_{self._grasped_prop_id}"
        )
        if bid < 0:
            return
        # Find the prop's freejoint qpos addr (first joint in the body chain)
        jadr = int(self._model.body_jntadr[bid])
        if jadr < 0:
            return
        qpos_addr = int(self._model.jnt_qposadr[jadr])
        tcp_pos = self._data.site_xpos[self._tcp_site_id]
        tcp_mat = self._data.site_xmat[self._tcp_site_id].reshape(3, 3)
        new_pos = tcp_pos + tcp_mat @ self._grasp_offset_pos
        tcp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, tcp_quat, self._grasp_offset_quat)
        # freejoint qpos layout: [px, py, pz, qw, qx, qy, qz]
        self._data.qpos[qpos_addr : qpos_addr + 3] = new_pos
        self._data.qpos[qpos_addr + 3 : qpos_addr + 7] = new_quat
        # Zero out velocity so it doesn't fight the kinematic override
        dofadr = int(self._model.jnt_dofadr[jadr])
        self._data.qvel[dofadr : dofadr + 6] = 0.0

    def _emit_joint_state(self) -> None:
        if self._event_sink is None:
            return
        joints = [float(self._data.qpos[i]) for i in self._arm_qpos_addrs]
        fingers = [float(self._data.qpos[i]) for i in self._finger_qpos_addrs]
        self._event_sink(
            {
                "type": "joint_state",
                "arm": joints,
                "fingers": fingers,
                "t": float(self._data.time),
            }
        )

    def _emit(self, kind: str, payload: dict) -> None:
        if self._event_sink is None:
            return
        self._event_sink({"type": kind, **payload, "t": float(self._data.time)})


def _make_assets_dict(mjcf_path: Path) -> dict[str, bytes]:
    """Pre-load mesh assets so from_xml_string can resolve them.

    The MJCF's compiler ``meshdir`` is ``meshes/visual``. We mount STL files
    from that directory under their bare filename (e.g. ``link_base.stl``)
    and also from sibling directories under a relative path (e.g.
    ``../gripper/base_link.stl``) so the gripper mesh references resolve.
    """
    assets: dict[str, bytes] = {}
    base = mjcf_path.parent / "meshes"
    visual_dir = base / "visual"
    for f in visual_dir.glob("*.stl"):
        assets[f.name] = f.read_bytes()
    # Sibling directories the MJCF references via `../<dir>/<file>`. Only
    # `gripper/` is referenced today; skip the collision/ alias dir which
    # repeats filenames found under visual/.
    for sib in ("gripper",):
        sib_dir = base / sib
        if not sib_dir.is_dir():
            continue
        for f in sib_dir.glob("*.stl"):
            assets[f"../{sib}/{f.name}"] = f.read_bytes()
    return assets


def _rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ-Euler → 3×3 rotation matrix. Used to derive the gripper-down axis
    from the public ``(rx, ry, rz)`` Pose parameters."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _rpy_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
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


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / max(np.linalg.norm(out), 1e-12)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    a = math.sin((1 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    return a * q0 + b * q1


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
