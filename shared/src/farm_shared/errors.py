from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class _Spec:
    code: int
    template: str
    docs_url_slug: str


class ErrorCode(Enum):
    """Structured error codes from DESIGN.md "Structured error catalog".

    Each member carries the numeric code, a str.format-style template, and the
    docs slug. ``format_error`` is the only sanctioned way to render one.
    """

    E1001 = _Spec(
        code=1001,
        template=(
            "No camera found at {device} — fix: 'farm doctor cameras', "
            "then 'farm config set camera.wrist.device /dev/videoN'"
        ),
        docs_url_slug="E1001",
    )
    E1002 = _Spec(
        code=1002,
        template=(
            "Calibration is {age_days} days old — fix: 'farm calibrate', "
            "or pass --accept-calibration"
        ),
        docs_url_slug="E1002",
    )
    E1003 = _Spec(
        code=1003,
        template=(
            "GPU container cold-starting (typical 8–25s). "
            "Holding the run open; arm will move when ready."
        ),
        docs_url_slug="E1003",
    )
    E1004 = _Spec(
        code=1004,
        template="API key rejected — fix: 'farm login', or check FARM_API_KEY env var",
        docs_url_slug="E1004",
    )
    E1005 = _Spec(
        code=1005,
        template=(
            "Dispatcher WebSocket dropped after {seconds} s. "
            "Auto-reconnecting; arm halted in place — fix: 'farm run --resume {run_id}'"
        ),
        docs_url_slug="E1005",
    )
    E1006 = _Spec(
        code=1006,
        template=(
            "Edge Agent v{agent_version} detected, Dispatcher requires v{required_version}+"
            " — fix: 'pip install -U farm-edge-agent'"
        ),
        docs_url_slug="E1006",
    )
    E1007 = _Spec(
        code=1007,
        template=(
            "Network probe FAILED: WebSocket upgrade blocked — fix: "
            "'farm doctor network' for diagnostics; try FARM_RELAY=on"
        ),
        docs_url_slug="E1007",
    )
    E1008 = _Spec(
        code=1008,
        template=(
            "Driver '{driver}' requires arm.ip in config — fix: "
            "'farm config set arm.ip <robot-ip>'"
        ),
        docs_url_slug="E1008",
    )
    E1009 = _Spec(
        code=1009,
        template="Config not found at {path} — fix: 'farm config init'",
        docs_url_slug="E1009",
    )
    E1010 = _Spec(
        code=1010,
        template="Required env var {name} is not set — fix: 'export {name}=...'",
        docs_url_slug="E1010",
    )
    E2001 = _Spec(
        code=2001,
        template=(
            "capability_card.action_space: '{value}' not in allowed set. "
            "Did you mean '{suggestion}'? "
            "Schema: https://farm.dev/schemas/capability_card.v1"
        ),
        docs_url_slug="E2001",
    )
    E3001 = _Spec(
        code=3001,
        template="Safety envelope violation: commanded pose outside workspace. Soft-stopped.",
        docs_url_slug="E3001",
    )
    E3002 = _Spec(
        code=3002,
        template="Watchdog timeout (>1s server silence). Arm halted in place.",
        docs_url_slug="E3002",
    )

    @property
    def code(self) -> int:
        return self.value.code

    @property
    def template(self) -> str:
        return self.value.template

    @property
    def docs_url_slug(self) -> str:
        return self.value.docs_url_slug


# Symbolic aliases for call sites that prefer named codes.
NO_CAMERA = ErrorCode.E1001
CALIBRATION_STALE = ErrorCode.E1002
GPU_COLD_START = ErrorCode.E1003
API_KEY_REJECTED = ErrorCode.E1004
WS_DROPPED = ErrorCode.E1005
VERSION_MISMATCH = ErrorCode.E1006
NETWORK_BLOCKED = ErrorCode.E1007
DRIVER_REQUIRES_ARM_IP = ErrorCode.E1008
CONFIG_NOT_FOUND = ErrorCode.E1009
ENV_VAR_MISSING = ErrorCode.E1010
CAPABILITY_CARD_INVALID = ErrorCode.E2001
SAFETY_ENVELOPE = ErrorCode.E3001
WATCHDOG_TIMEOUT = ErrorCode.E3002


def format_error(code: ErrorCode, **slots: Any) -> str:
    """Render an error as ``[FARM-Exxxx] <one-line> — fix: <action>``."""
    body = code.template.format(**slots)
    return f"[FARM-{code.name}] {body}"
