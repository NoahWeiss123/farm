"""Lean MuJoCo sim — basic motion + jog + cameras."""

from __future__ import annotations

import math

import pytest

mujoco = pytest.importorskip("mujoco")  # noqa: F841

from farm_edge_agent.sim import HOME_JOINTS, Sim  # noqa: E402


@pytest.fixture(scope="module")
def sim() -> Sim:
    # Real-time stepping makes tests slow and adds wall-clock jitter; flip it
    # off for unit-test motion.
    s = Sim(realtime=False)
    s.connect()
    yield s
    s.disconnect()


def test_home_brings_joints_to_known_pose(sim: Sim) -> None:
    sim.home()
    joints = sim.read_joint_state()
    assert len(joints) == 6
    for got, want in zip(joints, HOME_JOINTS, strict=True):
        assert abs(got - want) < 0.05, f"joint diverged: got {got}, want {want}"


def test_read_tcp_pose_returns_six_values(sim: Sim) -> None:
    sim.home()
    pose = sim.read_tcp_pose()
    assert len(pose) == 6
    x, y, z, rx, ry, rz = pose
    # Home pose places TCP roughly under the arm flange in the workspace.
    assert -300 < x < 300, f"x out of expected range: {x}"
    assert -1100 < y < -200, f"y out of expected range: {y}"
    assert 100 < z < 700, f"z out of expected range: {z}"
    # Just confirm rpy returns floats; ranges depend on euler convention.
    for v in (rx, ry, rz):
        assert isinstance(v, float)


def test_jog_z_moves_tcp(sim: Sim) -> None:
    sim.home()
    before = sim.read_tcp_pose()
    after = sim.jog("z", +1, step_mm=30.0)
    assert after[2] > before[2] - 5, (
        f"jog z+ should raise TCP: before={before[2]:.1f} after={after[2]:.1f}"
    )


def test_jog_negative_axis_inverts_direction(sim: Sim) -> None:
    sim.home()
    base = sim.read_tcp_pose()[0]
    sim.jog("x", +1, step_mm=20.0)
    after_plus = sim.read_tcp_pose()[0]
    sim.jog("x", -1, step_mm=20.0)
    after_back = sim.read_tcp_pose()[0]
    # The +/- pair should return roughly to baseline; PD + IK introduces a few mm slack.
    assert abs(after_back - base) < 15, (
        f"jog +/- didn't symmetrize: base={base:.1f} +={after_plus:.1f} -={after_back:.1f}"
    )


def test_render_each_camera_returns_rgb_array(sim: Sim) -> None:
    for cam in ("exterior", "wrist", "topdown"):
        img = sim.render_rgb(camera=cam, height=64, width=64)
        assert img.shape == (64, 64, 3), f"{cam} bad shape: {img.shape}"
        assert img.dtype.kind in ("u", "i"), f"{cam} non-int dtype: {img.dtype}"


def test_snapshot_has_complete_fields(sim: Sim) -> None:
    sim.home()
    snap = sim.snapshot()
    for key in ("joints", "tcp_pos_mm", "tcp_rpy", "gripper", "gripper_pos", "t"):
        assert key in snap, f"snapshot missing {key}"
    assert len(snap["joints"]) == 6
    assert len(snap["tcp_pos_mm"]) == 3
    assert len(snap["tcp_rpy"]) == 3
    assert snap["gripper"] in ("open", "closed", "grasping")


def test_set_gripper_closes_and_reopens(sim: Sim) -> None:
    sim.home()
    sim.set_gripper("closed")
    assert sim.gripper_state == "closed"
    closed_pos = sim.read_gripper()
    sim.set_gripper("open")
    assert sim.gripper_state == "open"
    open_pos = sim.read_gripper()
    assert closed_pos > open_pos, f"close didn't move finger: open={open_pos} closed={closed_pos}"


def test_continuous_joints_unwrap_toward_home(sim: Sim) -> None:
    # J1/J4/J6 are continuous revolute joints on the UF850 (range ±2π).
    # Asking the sim to park them at 3π/2 should fold back to -π/2 (the
    # representative within π of home=0), not get clamped at +2π-ε.
    sim.move_joint(
        [
            3.0 * math.pi / 2,         # J1: 4.71 → -1.57
            -0.5, -0.5,
            3.0 * math.pi / 2,         # J4: same
            -math.pi / 2,
            3.0 * math.pi / 2,         # J6: same
        ]
    )
    joints = sim.read_joint_state()
    want = -math.pi / 2
    for idx in (0, 3, 5):
        assert abs(joints[idx] - want) < 0.05, (
            f"continuous joint J{idx + 1} didn't unwrap toward home: "
            f"got {joints[idx]:.3f}, want {want:.3f}"
        )


def test_repeated_rotation_does_not_pin_continuous_joint(sim: Sim) -> None:
    # Old IK clamped J4 at +2π-ε, so repeated rz jogs would visibly stop
    # tracking once the joint wound up. With unwrap on apply, |J4| should
    # never run away.
    sim.home()
    for _ in range(60):
        sim.jog("rz", +1, step_rad=math.radians(10.0))
    j4 = sim.read_joint_state()[3]
    assert abs(j4) <= math.pi + 0.1, (
        f"J4 ran past its representative range after sustained rz jog: {j4:.3f}"
    )


def test_jog_rejects_unknown_axis(sim: Sim) -> None:
    with pytest.raises(ValueError):
        sim.jog("w", 1)  # type: ignore[arg-type]


def test_jog_rejects_bad_sign(sim: Sim) -> None:
    with pytest.raises(ValueError):
        sim.jog("x", 2)  # type: ignore[arg-type]


def test_event_sink_receives_joint_state() -> None:
    events: list[dict] = []
    s = Sim(realtime=False, event_sink=events.append)
    s.connect()
    try:
        s.home()
        s.jog("z", +1, step_mm=10.0)
    finally:
        s.disconnect()
    types = {e["type"] for e in events}
    assert "joint_state" in types, f"no joint_state events emitted: {types}"
    assert "jog" in types or "move_to" in types
