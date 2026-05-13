from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import IO, Any, Callable

DRIVERS: tuple[str, ...] = ("xarm", "franka", "lerobot-mock", "lerobot-so")


@dataclass
class RealArmConfig:
    driver: str | None = None
    arm_ip: str | None = None
    e_stop_ok: bool = False
    wrist_camera: str | None = None
    overhead_camera: str | None = None
    calibration_run: bool = False
    aborted: bool = False
    abort_step: str | None = None
    abort_reason: str | None = None


ConfigWriter = Callable[[dict[str, Any]], None]


def _prompt(
    stream_in: IO[str],
    stream_out: IO[str],
    prompt: str,
    default: str | None = None,
) -> str:
    suffix = f" [{default}]" if default else ""
    stream_out.write(f"{prompt}{suffix}: ")
    stream_out.flush()
    line = stream_in.readline()
    if line == "":
        return default if default is not None else ""
    answer = line.rstrip("\r\n")
    if not answer and default is not None:
        return default
    return answer


def _confirm(
    stream_in: IO[str],
    stream_out: IO[str],
    prompt: str,
    default: bool = True,
) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = _prompt(
        stream_in,
        stream_out,
        f"{prompt} ({suffix})",
        default="y" if default else "n",
    )
    return answer.strip().lower().startswith("y")


def _emit(config_writer: ConfigWriter | None, cfg: RealArmConfig) -> None:
    if config_writer is None:
        return
    config_writer(asdict(cfg))


def run_real_arm(
    stream_in: IO[str],
    stream_out: IO[str],
    config_writer: ConfigWriter | None = None,
) -> RealArmConfig:
    cfg = RealArmConfig()

    stream_out.write("Real-arm setup walk-through.\n")
    driver = _prompt(
        stream_in,
        stream_out,
        f"Driver? (one of: {', '.join(DRIVERS)})",
        default="xarm",
    )
    if driver not in DRIVERS:
        stream_out.write(f"unknown driver: {driver}\n")
        cfg.aborted = True
        cfg.abort_step = "driver"
        cfg.abort_reason = f"driver '{driver}' not in {DRIVERS}"
        _emit(config_writer, cfg)
        return cfg
    cfg.driver = driver
    _emit(config_writer, cfg)

    if driver != "lerobot-mock":
        ip = _prompt(stream_in, stream_out, "Arm IP", default="192.168.1.213")
        if not ip:
            stream_out.write("Arm IP is required for hardware drivers.\n")
            cfg.aborted = True
            cfg.abort_step = "arm_ip"
            cfg.abort_reason = "no arm IP provided"
            _emit(config_writer, cfg)
            return cfg
        cfg.arm_ip = ip
        _emit(config_writer, cfg)

    estop_ok = _confirm(
        stream_in,
        stream_out,
        "E-stop pressed and circuit verified?",
        default=False,
    )
    cfg.e_stop_ok = estop_ok
    if not estop_ok:
        stream_out.write("E-stop not verified; aborting.\n")
        cfg.aborted = True
        cfg.abort_step = "e_stop"
        cfg.abort_reason = "operator declined e-stop verification"
        _emit(config_writer, cfg)
        return cfg
    _emit(config_writer, cfg)

    wrist = _prompt(
        stream_in, stream_out, "Wrist camera device", default="/dev/video0"
    )
    cfg.wrist_camera = wrist
    overhead = _prompt(
        stream_in,
        stream_out,
        "Overhead camera device (blank for none)",
        default="",
    )
    cfg.overhead_camera = overhead or None
    _emit(config_writer, cfg)

    run_calib = _confirm(
        stream_in, stream_out, "Run calibration now?", default=True
    )
    cfg.calibration_run = run_calib
    _emit(config_writer, cfg)

    stream_out.write("Real-arm doctor complete.\n")
    return cfg
