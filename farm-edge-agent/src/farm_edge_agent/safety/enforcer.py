"""Composed safety boundary used by the control loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import ActionChunk, CheckResult, Pose, SafetyEvent
from .calibration import CalibrationCheck, CalibrationStatus
from .envelope import Envelope
from .estop import EstopCheck
from .singularity import SingularityCheck
from .velocity import VelocityCap

EventSink = Callable[[SafetyEvent], None]


@dataclass
class StartResult:
    """Outcome of the pre-run gate."""

    ok: bool
    calibration: CalibrationStatus | None
    events: list[SafetyEvent]


@dataclass
class ChunkResult:
    """Outcome of running a chunk through the per-step gates."""

    ok: bool
    chunk: ActionChunk
    was_clamped: bool
    events: list[SafetyEvent]


class SafetyEnforcer:
    """Single entry point for every safety gate in the Edge Agent.

    The control loop calls `pre_run()` once before a run begins (e-stop +
    calibration) and `check_chunk()` for each outbound action chunk
    (envelope, velocity clamp, singularity). Every event is also pushed to
    `sink` so the run record can capture it without a direct import.

    `halted` flips to True the first time any check produces a `violation`
    event. The control loop polls this and halts the arm at the next chunk
    boundary.
    """

    def __init__(
        self,
        *,
        envelope: Envelope,
        velocity: VelocityCap,
        singularity: SingularityCheck,
        estop: EstopCheck,
        calibration: CalibrationCheck,
        sink: EventSink | None = None,
    ) -> None:
        self._envelope = envelope
        self._velocity = velocity
        self._singularity = singularity
        self._estop = estop
        self._calibration = calibration
        self._sink = sink
        self._halted = False

    @property
    def halted(self) -> bool:
        return self._halted

    def _emit(self, event: SafetyEvent) -> None:
        if event.severity == "violation":
            self._halted = True
        if self._sink is not None:
            self._sink(event)

    def pre_run(self) -> StartResult:
        events: list[SafetyEvent] = []

        estop_result = self._estop.check()
        if estop_result.event is not None:
            events.append(estop_result.event)
            self._emit(estop_result.event)

        cal_result, cal_status = self._calibration.check()
        if cal_result.event is not None:
            events.append(cal_result.event)
            self._emit(cal_result.event)

        ok = estop_result.ok and cal_result.ok
        return StartResult(ok=ok, calibration=cal_status, events=events)

    def check_chunk(self, chunk: ActionChunk) -> ChunkResult:
        events: list[SafetyEvent] = []

        clamped, was_clamped = self._velocity.clamp(chunk)
        if was_clamped:
            event = SafetyEvent(
                kind="velocity_clamp",
                severity="warning",
                code="FARM-E3005",
                message="action chunk clamped to velocity cap",
            )
            events.append(event)
            self._emit(event)

        for waypoint in clamped.tcp_waypoints:
            result = self._check_waypoint(waypoint)
            if result.event is not None:
                events.append(result.event)
                self._emit(result.event)
            if not result.ok:
                return ChunkResult(
                    ok=False, chunk=clamped, was_clamped=was_clamped, events=events
                )

        return ChunkResult(
            ok=True, chunk=clamped, was_clamped=was_clamped, events=events
        )

    def _check_waypoint(self, pose: Pose) -> CheckResult:
        envelope_result = self._envelope.check(pose)
        if not envelope_result.ok:
            return envelope_result
        singularity_result = self._singularity.check(pose)
        if not singularity_result.ok:
            return singularity_result
        return CheckResult.pass_()
