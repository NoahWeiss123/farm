from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from farm_edge_agent.safety import ActionChunk, Pose, SafetyEvent
from farm_edge_agent.safety.calibration import STALE_AFTER_S, CalibrationCheck
from farm_edge_agent.safety.enforcer import SafetyEnforcer
from farm_edge_agent.safety.envelope import Envelope
from farm_edge_agent.safety.estop import EstopCheck
from farm_edge_agent.safety.singularity import SingularityCheck
from farm_edge_agent.safety.velocity import VelocityCap


@dataclass
class FakeDriver:
    armed: bool = True
    unreachable: set[tuple[float, ...]] = field(default_factory=set)
    colliding: set[tuple[float, ...]] = field(default_factory=set)

    def is_estop_armed(self) -> bool:
        return self.armed

    def check_pose_reachable(self, pose: Pose) -> bool:
        return tuple(pose) not in self.unreachable

    def check_self_collision(self, pose: Pose) -> bool:
        return tuple(pose) in self.colliding


def make_enforcer(
    tmp_path: Path,
    *,
    driver: FakeDriver,
    sink: list[SafetyEvent],
    calib_mtime: float = 1000.0,
    now: float = 1060.0,
    accept_calibration: bool = False,
) -> SafetyEnforcer:
    calib = tmp_path / "calibration.yaml"
    calib.write_bytes(b"data\n")
    os.utime(calib, (calib_mtime, calib_mtime))
    return SafetyEnforcer(
        envelope=Envelope(min_xyz=(-0.2, -0.2, 0.0), max_xyz=(0.2, 0.2, 0.4)),
        velocity=VelocityCap(joint_max=1.0, tcp_max_mps=0.25),
        singularity=SingularityCheck(driver),
        estop=EstopCheck(driver),
        calibration=CalibrationCheck(
            calib,
            accept_calibration=accept_calibration,
            now=lambda: now,
        ),
        sink=sink.append,
    )


def test_pre_run_fails_when_estop_unarmed(tmp_path: Path) -> None:
    sink: list[SafetyEvent] = []
    enf = make_enforcer(tmp_path, driver=FakeDriver(armed=False), sink=sink)

    start = enf.pre_run()

    assert not start.ok
    assert enf.halted
    assert any(e.kind == "estop_not_armed" for e in sink)


def test_pre_run_includes_calibration_hash(tmp_path: Path) -> None:
    sink: list[SafetyEvent] = []
    enf = make_enforcer(tmp_path, driver=FakeDriver(armed=True), sink=sink)

    start = enf.pre_run()

    assert start.ok
    assert start.calibration is not None
    assert len(start.calibration.sha256) == 64
    assert not enf.halted


def test_check_chunk_clamps_and_emits_warning(tmp_path: Path) -> None:
    sink: list[SafetyEvent] = []
    enf = make_enforcer(tmp_path, driver=FakeDriver(), sink=sink)
    chunk = ActionChunk(
        joint_velocities=[[2.0]],
        tcp_waypoints=[Pose(0.0, 0.0, 0.1)],
    )

    out = enf.check_chunk(chunk)

    assert out.ok
    assert out.was_clamped
    assert not enf.halted
    assert any(e.kind == "velocity_clamp" for e in sink)


def test_envelope_violation_halts_loop(tmp_path: Path) -> None:
    sink: list[SafetyEvent] = []
    enf = make_enforcer(tmp_path, driver=FakeDriver(), sink=sink)
    chunk = ActionChunk(tcp_waypoints=[Pose(0.5, 0.0, 0.1)])

    out = enf.check_chunk(chunk)

    assert not out.ok
    assert enf.halted
    assert any(e.kind == "envelope_violation" for e in sink)


def test_singularity_violation_halts_loop(tmp_path: Path) -> None:
    sink: list[SafetyEvent] = []
    bad_pose = Pose(0.1, 0.1, 0.1)
    driver = FakeDriver(unreachable={tuple(bad_pose)})
    enf = make_enforcer(tmp_path, driver=driver, sink=sink)

    out = enf.check_chunk(ActionChunk(tcp_waypoints=[bad_pose]))

    assert not out.ok
    assert enf.halted
    assert any(e.kind == "singularity_rejected" for e in sink)


def test_stale_calibration_blocks_until_accepted(tmp_path: Path) -> None:
    sink: list[SafetyEvent] = []
    refused = make_enforcer(
        tmp_path,
        driver=FakeDriver(),
        sink=sink,
        calib_mtime=0.0,
        now=STALE_AFTER_S * 3,
    )

    assert not refused.pre_run().ok
    assert refused.halted

    accepted_sink: list[SafetyEvent] = []
    accepted = make_enforcer(
        tmp_path,
        driver=FakeDriver(),
        sink=accepted_sink,
        calib_mtime=0.0,
        now=STALE_AFTER_S * 3,
        accept_calibration=True,
    )
    start = accepted.pre_run()
    assert start.ok
    assert not accepted.halted
    assert any(
        e.severity == "warning" and e.kind == "calibration_stale" for e in accepted_sink
    )
