from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from farm_edge_agent.config.doctor import Severity, check
from farm_edge_agent.config.schema import (
    ArmConfig,
    CameraConfig,
    CameraView,
    Config,
)
from farm_shared.errors import ErrorCode


def _config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = dict(
        api_key="sk-test",
        driver="lerobot-mock",
        camera=CameraConfig(wrist=CameraView(device="/dev/video0")),
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture
def fake_device(tmp_path: Path) -> Path:
    device = tmp_path / "video0"
    device.touch()
    return device


def test_valid_config_has_no_findings(fake_device: Path) -> None:
    cfg = _config(camera=CameraConfig(wrist=CameraView(device=str(fake_device))))
    assert check(cfg) == []


def test_missing_api_key_is_critical(fake_device: Path) -> None:
    cfg = _config(
        api_key="",
        camera=CameraConfig(wrist=CameraView(device=str(fake_device))),
    )
    findings = check(cfg)
    assert any(
        f.severity is Severity.CRITICAL and f.code is ErrorCode.E1004
        for f in findings
    )


def test_unexpanded_api_key_is_critical(fake_device: Path) -> None:
    cfg = _config(
        api_key="${FARM_API_KEY}",
        camera=CameraConfig(wrist=CameraView(device=str(fake_device))),
    )
    findings = check(cfg)
    assert any(f.code is ErrorCode.E1004 for f in findings)


def test_xarm_without_arm_ip_is_critical(fake_device: Path) -> None:
    cfg = _config(
        driver="xarm",
        arm=ArmConfig(ip=None),
        camera=CameraConfig(wrist=CameraView(device=str(fake_device))),
    )
    findings = check(cfg)
    arm_findings = [f for f in findings if f.code is ErrorCode.E1008]
    assert len(arm_findings) == 1
    assert arm_findings[0].severity is Severity.CRITICAL
    assert "arm.ip" in arm_findings[0].fix


def test_xarm_with_arm_ip_is_ok(fake_device: Path) -> None:
    cfg = _config(
        driver="xarm",
        arm=ArmConfig(ip="192.168.1.213"),
        camera=CameraConfig(wrist=CameraView(device=str(fake_device))),
    )
    assert all(f.code is not ErrorCode.E1008 for f in check(cfg))


def test_camera_device_missing_is_flagged() -> None:
    cfg = _config(camera=CameraConfig(wrist=CameraView(device="/dev/nope-12345")))
    findings = check(cfg)
    cam_findings = [f for f in findings if f.code is ErrorCode.E1001]
    assert len(cam_findings) == 1
    assert "/dev/nope-12345" in cam_findings[0].message


def test_stale_calibration_is_flagged(tmp_path: Path, fake_device: Path) -> None:
    calib = tmp_path / "wrist.yaml"
    calib.write_text("placeholder: true\n")
    old = time.time() - 2 * 86400
    os.utime(calib, (old, old))
    cfg = _config(
        camera=CameraConfig(
            wrist=CameraView(device=str(fake_device), intrinsics=calib)
        )
    )
    findings = check(cfg)
    stale = [f for f in findings if f.code is ErrorCode.E1002]
    assert len(stale) == 1
    assert "days old" in stale[0].message


def test_fresh_calibration_not_flagged(tmp_path: Path, fake_device: Path) -> None:
    calib = tmp_path / "wrist.yaml"
    calib.write_text("placeholder: true\n")
    cfg = _config(
        camera=CameraConfig(
            wrist=CameraView(device=str(fake_device), intrinsics=calib)
        )
    )
    assert all(f.code is not ErrorCode.E1002 for f in check(cfg))


def test_every_finding_has_error_code_and_fix() -> None:
    cfg = _config(
        api_key="",
        driver="xarm",
        camera=CameraConfig(wrist=CameraView(device="/dev/nope")),
    )
    findings = check(cfg)
    assert findings, "expected several findings for a broken config"
    for f in findings:
        assert isinstance(f.code, ErrorCode)
        assert f.code.name.startswith("E")
        assert f.fix
