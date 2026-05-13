from __future__ import annotations

import io
from typing import Any

from farm_edge_agent.doctor import real_arm


def _streams(script: str) -> tuple[io.StringIO, io.StringIO]:
    return io.StringIO(script), io.StringIO()


def _capture_writer() -> tuple[list[dict[str, Any]], real_arm.ConfigWriter]:
    snapshots: list[dict[str, Any]] = []

    def writer(payload: dict[str, Any]) -> None:
        snapshots.append(dict(payload))

    return snapshots, writer


def test_full_xarm_flow_records_every_answer() -> None:
    script = "\n".join(
        [
            "xarm",            # driver
            "192.168.1.50",    # arm IP
            "y",               # e-stop confirmed
            "/dev/video0",     # wrist camera
            "/dev/video1",     # overhead camera
            "y",               # run calibration
            "",
        ]
    )
    stream_in, stream_out = _streams(script)
    snapshots, writer = _capture_writer()
    cfg = real_arm.run_real_arm(stream_in, stream_out, config_writer=writer)

    assert cfg.driver == "xarm"
    assert cfg.arm_ip == "192.168.1.50"
    assert cfg.e_stop_ok is True
    assert cfg.wrist_camera == "/dev/video0"
    assert cfg.overhead_camera == "/dev/video1"
    assert cfg.calibration_run is True
    assert cfg.aborted is False
    assert "complete" in stream_out.getvalue().lower()
    assert len(snapshots) >= 4
    assert snapshots[-1]["driver"] == "xarm"
    assert snapshots[-1]["calibration_run"] is True


def test_mock_driver_skips_arm_ip_prompt() -> None:
    script = "\n".join(
        [
            "lerobot-mock",  # driver
            "y",             # e-stop
            "/dev/video0",   # wrist
            "",              # no overhead
            "n",             # skip calibration
            "",
        ]
    )
    stream_in, stream_out = _streams(script)
    cfg = real_arm.run_real_arm(stream_in, stream_out)

    assert cfg.driver == "lerobot-mock"
    assert cfg.arm_ip is None
    assert cfg.e_stop_ok is True
    assert cfg.wrist_camera == "/dev/video0"
    assert cfg.overhead_camera is None
    assert cfg.calibration_run is False
    assert cfg.aborted is False
    out = stream_out.getvalue()
    assert "Arm IP" not in out


def test_unknown_driver_aborts() -> None:
    stream_in, stream_out = _streams("not-a-driver\n")
    snapshots, writer = _capture_writer()
    cfg = real_arm.run_real_arm(stream_in, stream_out, config_writer=writer)

    assert cfg.aborted is True
    assert cfg.abort_step == "driver"
    assert cfg.driver is None
    assert "unknown driver" in stream_out.getvalue()
    assert snapshots[-1]["aborted"] is True


def test_estop_decline_aborts_before_camera_prompts() -> None:
    script = "\n".join(["xarm", "192.168.1.50", "n", ""])
    stream_in, stream_out = _streams(script)
    cfg = real_arm.run_real_arm(stream_in, stream_out)

    assert cfg.aborted is True
    assert cfg.abort_step == "e_stop"
    assert cfg.e_stop_ok is False
    assert cfg.wrist_camera is None
    out = stream_out.getvalue()
    assert "E-stop not verified" in out
    assert "Wrist camera" not in out


def test_eof_uses_defaults() -> None:
    stream_in, stream_out = _streams("")
    cfg = real_arm.run_real_arm(stream_in, stream_out)

    # default driver xarm, no arm ip (empty default for that prompt) -> abort on arm ip
    assert cfg.driver == "xarm"
    assert cfg.aborted is True
    assert cfg.abort_step in {"arm_ip", "e_stop"}


def test_config_writer_called_incrementally() -> None:
    script = "\n".join(
        [
            "xarm",
            "10.0.0.1",
            "y",
            "/dev/video0",
            "",
            "y",
            "",
        ]
    )
    stream_in, stream_out = _streams(script)
    snapshots, writer = _capture_writer()
    real_arm.run_real_arm(stream_in, stream_out, config_writer=writer)

    drivers = [s["driver"] for s in snapshots]
    arm_ips = [s["arm_ip"] for s in snapshots]
    assert drivers[0] == "xarm"
    assert arm_ips.count("10.0.0.1") >= 1
    assert snapshots[-1]["wrist_camera"] == "/dev/video0"
    assert snapshots[-1]["calibration_run"] is True
