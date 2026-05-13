from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

from farm_edge_agent.doctor import cameras


def test_default_lister_picks_macos_on_darwin() -> None:
    fn = cameras.default_lister(system="Darwin")
    assert fn is cameras.list_macos_devices


def test_default_lister_picks_linux_otherwise() -> None:
    assert cameras.default_lister(system="Linux") is cameras.list_linux_devices
    assert cameras.default_lister(system="Windows") is cameras.list_linux_devices


def test_list_macos_devices_returns_avf_indices() -> None:
    devices = cameras.list_macos_devices(max_index=3)
    assert devices == ["AVF:0", "AVF:1", "AVF:2"]


def test_enumerate_uses_injected_lister_and_prober() -> None:
    fake_props = cameras.DeviceProperties(width=1280, height=720, fps=30.0)
    captured_calls: list[str] = []

    def lister() -> list[str]:
        return ["/dev/video0", "/dev/video2"]

    def prober(path: str) -> cameras.DeviceProperties | None:
        captured_calls.append(path)
        return fake_props if path == "/dev/video0" else None

    def calib(path: str) -> datetime | None:
        return datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc) if path == "/dev/video0" else None

    out = cameras.enumerate_cameras(
        lister=lister, prober=prober, calibration_lookup=calib
    )
    assert captured_calls == ["/dev/video0", "/dev/video2"]
    assert len(out) == 2
    assert out[0].path == "/dev/video0"
    assert out[0].properties == fake_props
    assert out[0].calibration_mtime == datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    assert out[0].error_code is None
    assert out[1].path == "/dev/video2"
    assert out[1].properties is None
    assert out[1].error_code == "FARM-E1001"


def test_format_camera_line_open_device_has_resolution_fps_calibration() -> None:
    info = cameras.CameraInfo(
        path="/dev/video0",
        properties=cameras.DeviceProperties(width=1920, height=1080, fps=29.97),
        calibration_mtime=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )
    line = cameras.format_camera_line(info)
    assert "/dev/video0" in line
    assert "1920x1080" in line
    assert "30.0fps" in line
    assert "2026-05-01" in line


def test_format_camera_line_unopened_device_carries_error_code() -> None:
    info = cameras.CameraInfo(
        path="/dev/video1",
        properties=None,
        calibration_mtime=None,
        error_code="FARM-E1001",
    )
    line = cameras.format_camera_line(info)
    assert "/dev/video1" in line
    assert "FARM-E1001" in line
    assert "not accessible" in line


def test_format_camera_line_no_calibration_shows_marker() -> None:
    info = cameras.CameraInfo(
        path="AVF:0",
        properties=cameras.DeviceProperties(width=640, height=480, fps=15.0),
        calibration_mtime=None,
    )
    line = cameras.format_camera_line(info)
    assert "AVF:0" in line
    assert "no calibration" in line


def test_run_writes_a_line_per_camera_to_stream() -> None:
    buf = io.StringIO()
    props = cameras.DeviceProperties(width=640, height=480, fps=30.0)

    def lister() -> list[str]:
        return ["/dev/video0"]

    def prober(_: str) -> cameras.DeviceProperties:
        return props

    def calib(_: str) -> None:
        return None

    out = cameras.run(out=buf, lister=lister, prober=prober, calibration_lookup=calib)
    text = buf.getvalue()
    assert len(out) == 1
    assert "/dev/video0" in text
    assert "640x480" in text


def test_run_emits_e1001_when_no_devices_found() -> None:
    buf = io.StringIO()

    def lister() -> list[str]:
        return []

    out = cameras.run(out=buf, lister=lister, prober=lambda p: None, calibration_lookup=lambda p: None)
    assert out == []
    assert "FARM-E1001" in buf.getvalue()


def test_default_calibration_lookup_returns_none_when_missing(tmp_path: Path) -> None:
    assert cameras.default_calibration_lookup("/dev/video0", calib_dir=tmp_path) is None


def test_default_calibration_lookup_returns_mtime_when_present(tmp_path: Path) -> None:
    f = tmp_path / "dev_video0.yaml"
    f.write_text("intrinsics: {}\n")
    mtime = cameras.default_calibration_lookup("/dev/video0", calib_dir=tmp_path)
    assert mtime is not None
    assert mtime.tzinfo is timezone.utc
