from __future__ import annotations

import glob
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Callable


@dataclass(frozen=True)
class DeviceProperties:
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class CameraInfo:
    path: str
    properties: DeviceProperties | None
    calibration_mtime: datetime | None
    error_code: str | None = None


def list_linux_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def list_macos_devices(max_index: int = 8) -> list[str]:
    return [f"AVF:{i}" for i in range(max_index)]


def default_lister(system: str | None = None) -> Callable[[], list[str]]:
    sys_name = (system or platform.system()).lower()
    if sys_name == "darwin":
        return list_macos_devices
    return list_linux_devices


def probe_with_opencv(path: str) -> DeviceProperties | None:
    import cv2

    if path.startswith("AVF:"):
        index = int(path.split(":", 1)[1])
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return DeviceProperties(width=width, height=height, fps=fps)
    finally:
        cap.release()


def default_calibration_lookup(
    path: str, calib_dir: Path | None = None
) -> datetime | None:
    base = calib_dir or Path.home() / ".farm" / "calibration"
    sanitized = path.replace("/", "_").replace(":", "_").strip("_")
    candidate = base / f"{sanitized}.yaml"
    if not candidate.exists():
        return None
    return datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)


def enumerate_cameras(
    lister: Callable[[], list[str]] | None = None,
    prober: Callable[[str], DeviceProperties | None] | None = None,
    calibration_lookup: Callable[[str], datetime | None] | None = None,
) -> list[CameraInfo]:
    _list = lister or default_lister()
    _probe = prober or probe_with_opencv
    _calib = calibration_lookup or default_calibration_lookup

    out: list[CameraInfo] = []
    for path in _list():
        props = _probe(path)
        mtime = _calib(path)
        err = None if props is not None else "FARM-E1001"
        out.append(
            CameraInfo(
                path=path,
                properties=props,
                calibration_mtime=mtime,
                error_code=err,
            )
        )
    return out


def format_camera_line(info: CameraInfo) -> str:
    if info.error_code is not None or info.properties is None:
        return (
            f"{info.path}  not accessible  "
            f"[{info.error_code or 'FARM-E1001'}]  "
            "fix: 'farm doctor cameras', then "
            "'farm config set camera.wrist.device /dev/videoN'"
        )
    res = f"{info.properties.width}x{info.properties.height}"
    fps = f"{info.properties.fps:.1f}fps"
    calib = (
        info.calibration_mtime.isoformat()
        if info.calibration_mtime is not None
        else "no calibration"
    )
    return f"{info.path}  {res}  {fps}  calibration: {calib}"


def run(
    out: IO[str] | None = None,
    lister: Callable[[], list[str]] | None = None,
    prober: Callable[[str], DeviceProperties | None] | None = None,
    calibration_lookup: Callable[[str], datetime | None] | None = None,
) -> list[CameraInfo]:
    stream = out if out is not None else sys.stdout
    cams = enumerate_cameras(
        lister=lister, prober=prober, calibration_lookup=calibration_lookup
    )
    if not cams:
        stream.write(
            "[FARM-E1001] No camera found at /dev/video0 — fix: 'farm doctor cameras', "
            "then 'farm config set camera.wrist.device /dev/videoN'\n"
        )
        return cams
    for cam in cams:
        stream.write(format_camera_line(cam) + "\n")
    return cams
