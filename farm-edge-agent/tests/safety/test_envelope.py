from farm_edge_agent.safety import Pose, SafetyEvent
from farm_edge_agent.safety.envelope import Envelope


def make_envelope(captured: list[SafetyEvent] | None = None) -> Envelope:
    return Envelope(
        min_xyz=(-0.2, -0.2, 0.0),
        max_xyz=(0.2, 0.2, 0.4),
        on_event=(captured.append if captured is not None else None),
    )


def test_pose_inside_envelope_passes() -> None:
    env = make_envelope()
    result = env.check(Pose(0.0, 0.0, 0.2))
    assert result.ok
    assert result.event is None


def test_pose_on_boundary_passes() -> None:
    env = make_envelope()
    assert env.check(Pose(-0.2, 0.2, 0.4)).ok


def test_pose_one_mm_outside_violates() -> None:
    captured: list[SafetyEvent] = []
    env = make_envelope(captured)
    result = env.check(Pose(0.201, 0.0, 0.2))
    assert not result.ok
    assert result.event is not None
    assert result.event.code == "FARM-E3001"
    assert result.event.kind == "envelope_violation"
    assert captured == [result.event]


def test_violation_includes_axis_and_bounds() -> None:
    env = make_envelope()
    result = env.check(Pose(0.0, 0.0, -0.001))
    assert result.event is not None
    detail = result.event.detail
    assert detail["axis"] == "z"
    assert detail["min"] == 0.0


def test_inverted_bounds_rejected() -> None:
    try:
        Envelope(min_xyz=(0.0, 0.0, 0.0), max_xyz=(-0.1, 0.0, 0.0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
