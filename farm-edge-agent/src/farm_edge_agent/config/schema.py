"""Pydantic models for ~/.farm/config.yaml.

Mirrors the example YAML in DESIGN.md → Developer-Facing Surface → Config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Driver = Literal["xarm", "franka", "lerobot-mock"]


class CameraView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str
    intrinsics: Path | None = None


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wrist: CameraView
    overhead: CameraView | None = None


class ArmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip: str | None = None


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_frames: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str
    workspace: str | None = None
    driver: Driver = "lerobot-mock"
    arm: ArmConfig = Field(default_factory=ArmConfig)
    camera: CameraConfig
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
