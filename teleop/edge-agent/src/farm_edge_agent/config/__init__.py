"""Config loader, schema, and doctor for the FARM Edge Agent."""

from __future__ import annotations

from pathlib import Path

from .doctor import Finding, Severity, check
from .loader import (
    ConfigError,
    ConfigNotFoundError,
    EnvVarMissingError,
    default_config_path,
    load_config,
    resolve_config_path,
)
from .schema import (
    ArmConfig,
    CameraConfig,
    CameraView,
    Config,
    TelemetryConfig,
)

_TEMPLATE_PATH = Path(__file__).parent / "template.yaml"


def read_template() -> str:
    """Return the contents of the bundled config template."""
    return _TEMPLATE_PATH.read_text()


__all__ = [
    "ArmConfig",
    "CameraConfig",
    "CameraView",
    "Config",
    "ConfigError",
    "ConfigNotFoundError",
    "EnvVarMissingError",
    "Finding",
    "Severity",
    "TelemetryConfig",
    "check",
    "default_config_path",
    "load_config",
    "read_template",
    "resolve_config_path",
]
