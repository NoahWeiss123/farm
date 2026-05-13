from farm_edge_agent.safety import ActionChunk, Pose
from farm_edge_agent.safety.velocity import VelocityCap


def test_chunk_within_caps_unclamped() -> None:
    cap = VelocityCap(joint_max=1.0, tcp_max_mps=0.25)
    chunk = ActionChunk(
        joint_velocities=[[0.5, -0.5, 0.0]],
        tcp_waypoints=[Pose(0.0, 0.0, 0.0), Pose(0.05, 0.0, 0.0)],
        duration_s=1.0,
    )
    out, was_clamped = cap.clamp(chunk)
    assert not was_clamped
    assert out.joint_velocities == [[0.5, -0.5, 0.0]]
    assert out.tcp_waypoints[1] == Pose(0.05, 0.0, 0.0)


def test_joint_velocity_clamped_above_cap() -> None:
    cap = VelocityCap(joint_max=1.0, tcp_max_mps=10.0)
    chunk = ActionChunk(joint_velocities=[[2.0, -3.0, 0.5]])
    out, was_clamped = cap.clamp(chunk)
    assert was_clamped
    assert out.joint_velocities[0] == [1.0, -1.0, 0.5]


def test_tcp_waypoint_speed_clamped() -> None:
    cap = VelocityCap(joint_max=10.0, tcp_max_mps=0.1)
    chunk = ActionChunk(
        tcp_waypoints=[Pose(0.0, 0.0, 0.0), Pose(1.0, 0.0, 0.0)],
        duration_s=1.0,
    )
    out, was_clamped = cap.clamp(chunk)
    assert was_clamped
    assert out.tcp_waypoints[1].x == 0.1


def test_invalid_caps_rejected() -> None:
    for joint_max, tcp_max in [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)]:
        try:
            VelocityCap(joint_max=joint_max, tcp_max_mps=tcp_max)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for invalid cap")
