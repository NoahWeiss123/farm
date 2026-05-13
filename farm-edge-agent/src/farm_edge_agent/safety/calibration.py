"""Calibration drift detection.

Hashes the calibration file at run start and refuses to begin when the file's
mtime is older than the staleness threshold unless the operator passes
`--accept-calibration`. The hash is surfaced so it can be written into the
run record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import CheckResult, SafetyEvent

STALE_AFTER_S = 24 * 60 * 60

NowFn = Callable[[], float]


@dataclass(frozen=True)
class CalibrationStatus:
    """Snapshot of a calibration file ready to be embedded in the run record."""

    path: str
    sha256: str
    mtime_s: float
    age_s: float
    accepted_stale: bool


class CalibrationCheck:
    """Hash-and-compare gate.

    `check()` returns a pass result whose `event` is `None` on the happy path,
    a `warning` event when stale but explicitly accepted, and a `violation`
    when stale without acceptance. The hash is always computed so callers can
    record it on every run.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        *,
        accept_calibration: bool = False,
        stale_after_s: float = STALE_AFTER_S,
        now: NowFn | None = None,
    ) -> None:
        self._path = Path(calibration_path)
        self._accept = accept_calibration
        self._stale_after = stale_after_s
        self._now = now

    def _current_time(self) -> float:
        if self._now is not None:
            return self._now()
        import time

        return time.time()

    def status(self) -> CalibrationStatus:
        data = self._path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        mtime = self._path.stat().st_mtime
        age = self._current_time() - mtime
        return CalibrationStatus(
            path=str(self._path),
            sha256=sha,
            mtime_s=mtime,
            age_s=age,
            accepted_stale=self._accept and age > self._stale_after,
        )

    def check(self) -> tuple[CheckResult, CalibrationStatus]:
        status = self.status()
        if status.age_s <= self._stale_after:
            return CheckResult.pass_(), status

        days = status.age_s / 86400
        if self._accept:
            event = SafetyEvent(
                kind="calibration_stale",
                severity="warning",
                code="FARM-E1002",
                message=(
                    f"calibration is {days:.1f} days old; "
                    "running anyway because --accept-calibration was passed"
                ),
                detail={
                    "path": status.path,
                    "sha256": status.sha256,
                    "age_s": status.age_s,
                    "accepted": True,
                },
            )
            return CheckResult(ok=True, event=event), status

        event = SafetyEvent(
            kind="calibration_stale",
            severity="violation",
            code="FARM-E1002",
            message=(
                f"calibration is {days:.1f} days old. "
                "fix: 'farm calibrate', or pass --accept-calibration"
            ),
            detail={
                "path": status.path,
                "sha256": status.sha256,
                "age_s": status.age_s,
                "accepted": False,
            },
        )
        return CheckResult.fail(event), status
