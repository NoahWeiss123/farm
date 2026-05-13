from __future__ import annotations

import difflib
import json
from importlib.resources import files
from typing import Any, Literal

from jsonschema import Draft7Validator
from pydantic import BaseModel, ConfigDict

ActionSpace = Literal[
    "joint_position_delta",
    "ee_pose_delta_base_frame",
    "ee_pose_delta_tool_frame",
    "ee_velocity",
]
Determinism = Literal["deterministic", "stochastic", "seeded"]
Role = Literal["planner", "controller", "critic"]

ACTION_SPACES: tuple[str, ...] = (
    "joint_position_delta",
    "ee_pose_delta_base_frame",
    "ee_pose_delta_tool_frame",
    "ee_velocity",
)
DETERMINISMS: tuple[str, ...] = ("deterministic", "stochastic", "seeded")

SCHEMA_URL = "https://farm.dev/schemas/capability_card.v1"


class CapabilityCardError(ValueError):
    """Raised when a capability card fails validation. Carries a fix-shaped message."""


_SCHEMA_CACHE: dict[str, Any] | None = None


def load_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        raw = files("farm_shared.schemas").joinpath("capability_card.v1.json").read_text()
        _SCHEMA_CACHE = json.loads(raw)
    return _SCHEMA_CACHE


class Embodiment(BaseModel):
    model_config = ConfigDict(extra="allow")

    arm: str
    dof: int
    action_space: ActionSpace
    control_rate_hz: float | None = None


class CapabilityCard(BaseModel):
    """The router-readable description of a backend.

    Matches the YAML structure in DESIGN.md "Backend types and capability cards".
    Loaded via :meth:`from_dict`, which runs JSON Schema validation first so
    enum mismatches surface as did-you-mean suggestions.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    roles: list[Role]
    embodiment: Embodiment
    input_modalities: list[str]
    camera_views: list[str] = []
    skills: list[dict[str, Any]]
    latency: dict[str, Any] | None = None
    cost_per_chunk_usd: float | None = None
    determinism: Determinism | None = None
    safety: dict[str, Any] | None = None
    fallbacks: list[str] = []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityCard:
        validator = Draft7Validator(load_schema())
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if errors:
            raise _to_friendly_error(errors[0])
        return cls.model_validate(data)


def _to_friendly_error(err: Any) -> CapabilityCardError:
    path = list(err.absolute_path)
    if path == ["embodiment", "action_space"] and err.validator == "enum":
        bad = err.instance
        match = difflib.get_close_matches(str(bad), ACTION_SPACES, n=1)
        suggestion = match[0] if match else "(no close match)"
        return CapabilityCardError(
            f"capability_card.action_space: '{bad}' not in allowed set. "
            f"Did you mean '{suggestion}'? Schema: {SCHEMA_URL}"
        )
    if path == ["determinism"] and err.validator == "enum":
        bad = err.instance
        match = difflib.get_close_matches(str(bad), DETERMINISMS, n=1)
        suggestion = match[0] if match else "(no close match)"
        return CapabilityCardError(
            f"capability_card.determinism: '{bad}' not in allowed set. "
            f"Did you mean '{suggestion}'? Schema: {SCHEMA_URL}"
        )
    location = ".".join(str(p) for p in path) or "<root>"
    return CapabilityCardError(f"capability_card.{location}: {err.message}")
