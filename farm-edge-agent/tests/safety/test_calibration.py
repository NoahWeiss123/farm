from __future__ import annotations

import hashlib
import os
from pathlib import Path

from farm_edge_agent.safety.calibration import STALE_AFTER_S, CalibrationCheck


def write_calib(path: Path, mtime: float, body: bytes = b"intrinsics: ok\n") -> None:
    path.write_bytes(body)
    os.utime(path, (mtime, mtime))


def test_fresh_calibration_passes(tmp_path: Path) -> None:
    calib = tmp_path / "calibration.yaml"
    body = b"focal_x: 600\n"
    write_calib(calib, mtime=1000.0, body=body)
    check = CalibrationCheck(calib, now=lambda: 1000.0 + 60)

    result, status = check.check()
    assert result.ok
    assert result.event is None
    assert status.sha256 == hashlib.sha256(body).hexdigest()
    assert status.accepted_stale is False


def test_stale_without_flag_refuses(tmp_path: Path) -> None:
    calib = tmp_path / "calibration.yaml"
    write_calib(calib, mtime=0.0)
    check = CalibrationCheck(calib, now=lambda: STALE_AFTER_S * 3)

    result, status = check.check()
    assert not result.ok
    assert result.event is not None
    assert result.event.code == "FARM-E1002"
    assert result.event.severity == "violation"
    assert status.age_s > STALE_AFTER_S


def test_stale_with_flag_warns_but_allows(tmp_path: Path) -> None:
    calib = tmp_path / "calibration.yaml"
    write_calib(calib, mtime=0.0)
    check = CalibrationCheck(
        calib,
        accept_calibration=True,
        now=lambda: STALE_AFTER_S * 3,
    )

    result, status = check.check()
    assert result.ok
    assert result.event is not None
    assert result.event.severity == "warning"
    assert status.accepted_stale is True


def test_status_hash_is_stable(tmp_path: Path) -> None:
    calib = tmp_path / "calibration.yaml"
    body = b"abc\n"
    write_calib(calib, mtime=1000.0, body=body)
    check = CalibrationCheck(calib, now=lambda: 1000.0)

    first = check.status()
    second = check.status()
    assert first.sha256 == second.sha256 == hashlib.sha256(body).hexdigest()
