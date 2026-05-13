import dataclasses

from farm_shared.run_state import RunState


def _sample() -> RunState:
    return RunState(
        joint_pose=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        tcp_pose=([0.30, -0.10, 0.25], [0.0, 0.0, 0.0, 1.0]),
        gripper_state="grasping",
        task_progress=2,
        last_completed_chunk=7,
        observation_snapshot={
            "wrist": "/tmp/run-x/wrist.jpg",
            "overhead": "/tmp/run-x/overhead.jpg",
        },
        critic_summary="picked the red block, on approach to stack",
    )


def test_roundtrip_through_asdict():
    state = _sample()
    d = dataclasses.asdict(state)
    d["tcp_pose"] = (d["tcp_pose"][0], d["tcp_pose"][1])
    restored = RunState(**d)
    assert restored == state


def test_critic_summary_optional():
    state = _sample()
    state_no_summary = dataclasses.replace(state, critic_summary=None)
    d = dataclasses.asdict(state_no_summary)
    d["tcp_pose"] = (d["tcp_pose"][0], d["tcp_pose"][1])
    restored = RunState(**d)
    assert restored.critic_summary is None
    assert restored == state_no_summary
