"""Validate a loaded Config and surface fixable findings."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from farm_shared.errors import ErrorCode

from .schema import CameraView, Config

CALIBRATION_MAX_AGE_SECONDS = 24 * 3600


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: ErrorCode
    message: str
    fix: str


def _iter_views(config: Config) -> list[tuple[str, CameraView]]:
    views: list[tuple[str, CameraView]] = [("wrist", config.camera.wrist)]
    if config.camera.overhead is not None:
        views.append(("overhead", config.camera.overhead))
    return views


def _check_api_key(config: Config) -> list[Finding]:
    if config.api_key and not config.api_key.startswith("${"):
        return []
    return [
        Finding(
            severity=Severity.CRITICAL,
            code=ErrorCode.E1004,
            message="api_key is empty or contains an unexpanded ${...} reference",
            fix="farm login, or export FARM_API_KEY",
        )
    ]


def _check_arm_ip(config: Config) -> list[Finding]:
    if config.driver == "lerobot-mock":
        return []
    if config.arm.ip:
        return []
    return [
        Finding(
            severity=Severity.CRITICAL,
            code=ErrorCode.E1008,
            message=f"driver '{config.driver}' requires arm.ip",
            fix="farm config set arm.ip <robot-ip>",
        )
    ]


def _check_cameras(config: Config) -> list[Finding]:
    findings: list[Finding] = []
    for name, view in _iter_views(config):
        if Path(view.device).exists():
            continue
        findings.append(
            Finding(
                severity=Severity.WARNING,
                code=ErrorCode.E1001,
                message=f"No camera found at {view.device}",
                fix=f"farm doctor cameras, then farm config set camera.{name}.device /dev/videoN",
            )
        )
    return findings


def _check_calibration(config: Config) -> list[Finding]:
    findings: list[Finding] = []
    now = time.time()
    for _, view in _iter_views(config):
        if view.intrinsics is None:
            continue
        path = Path(view.intrinsics)
        if not path.exists():
            continue
        age = now - path.stat().st_mtime
        if age <= CALIBRATION_MAX_AGE_SECONDS:
            continue
        days = age / 86400
        findings.append(
            Finding(
                severity=Severity.WARNING,
                code=ErrorCode.E1002,
                message=f"Calibration is {days:.1f} days old ({path})",
                fix="farm calibrate, or pass --accept-calibration",
            )
        )
    return findings


def check(config: Config) -> list[Finding]:
    """Validate `config` and return a list of fixable findings.

    Findings are ordered: api_key → arm → camera → calibration.
    Critical severity means `farm config doctor` exits non-zero.
    """
    findings: list[Finding] = []
    findings.extend(_check_api_key(config))
    findings.extend(_check_arm_ip(config))
    findings.extend(_check_cameras(config))
    findings.extend(_check_calibration(config))
    return findings
