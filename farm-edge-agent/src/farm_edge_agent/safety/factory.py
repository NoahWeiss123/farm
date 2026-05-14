"""Sim-default SafetyEnforcer construction.

Phase 3 binding glue. Real-arm runs will supply different envelope/velocity
caps + a real calibration file via config; the sim integration tests and
the local CLI smoke-test get sensible defaults from here so the RunLoop
doesn't have to know about them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from farm_edge_agent.safety import SafetyEvent
from farm_edge_agent.safety.calibration import CalibrationCheck
from farm_edge_agent.safety.enforcer import SafetyEnforcer
from farm_edge_agent.safety.envelope import Envelope
from farm_edge_agent.safety.estop import EstopCheck
from farm_edge_agent.safety.singularity import SingularityCheck
from farm_edge_agent.safety.velocity import VelocityCap

# Generous AABB in meters, in the arm base frame. Covers the demo workspace
# (table centered at y=-0.7, top z=0.265) with margin so legitimate pre-grasp
# poses pass through. Tighten when running on real hardware.
SIM_ENVELOPE_MIN_M = (-0.40, -1.05, -0.05)
SIM_ENVELOPE_MAX_M = (0.40, -0.10, 1.20)
SIM_TCP_MAX_MPS = 0.30
SIM_JOINT_MAX_RPS = 3.14


def make_sim_enforcer(
    driver: object,
    *,
    calibration_path: Path | None = None,
    accept_stale_calibration: bool = True,
    sink: Callable[[SafetyEvent], None] | None = None,
) -> SafetyEnforcer:
    """Build a SafetyEnforcer suitable for the local sim path.

    A calibration file is created on demand if one isn't supplied so the
    CalibrationCheck has something to hash; the file is set to ``mtime
    now`` so it's never stale.
    """
    if calibration_path is None:
        calibration_path = Path.home() / ".farm" / "calibration" / "sim_default.toml"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    if not calibration_path.exists():
        calibration_path.write_text(
            "# placeholder sim calibration; replace with real hand-eye on hardware\n"
            f"created_at = {time.time():.0f}\n"
        )
    return SafetyEnforcer(
        envelope=Envelope(SIM_ENVELOPE_MIN_M, SIM_ENVELOPE_MAX_M),
        velocity=VelocityCap(joint_max=SIM_JOINT_MAX_RPS, tcp_max_mps=SIM_TCP_MAX_MPS),
        singularity=SingularityCheck(driver=driver),  # type: ignore[arg-type]
        estop=EstopCheck(driver=driver),  # type: ignore[arg-type]
        calibration=CalibrationCheck(
            calibration_path,
            accept_calibration=accept_stale_calibration,
        ),
        sink=sink,
    )


__all__ = ["make_sim_enforcer", "SIM_ENVELOPE_MIN_M", "SIM_ENVELOPE_MAX_M"]
