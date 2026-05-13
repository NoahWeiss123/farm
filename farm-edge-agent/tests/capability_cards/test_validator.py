"""Tests for capability_cards.validator."""

from pathlib import Path
from typing import Any

import yaml
from farm_edge_agent.capability_cards.validator import validate

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_valid_card_has_no_findings() -> None:
    assert validate(_load("valid_pi05.yaml")) == []


def test_invalid_action_space_suggests_correct_value() -> None:
    findings = validate(_load("invalid_action_space.yaml"))
    enum_findings = [f for f in findings if f.did_you_mean is not None]
    assert len(enum_findings) == 1
    f = enum_findings[0]
    assert f.value == "ee_pose_delta"
    assert f.did_you_mean == "ee_pose_delta_base_frame"
    assert "embodiment.action_space" in f.path


def test_invalid_action_space_message_matches_design() -> None:
    findings = validate(_load("invalid_action_space.yaml"))
    enum = next(f for f in findings if f.did_you_mean is not None)
    assert enum.message == (
        "capability_card.embodiment.action_space: "
        "'ee_pose_delta' not in allowed set. "
        "Did you mean 'ee_pose_delta_base_frame'? "
        "Schema: https://farm.dev/schemas/capability_card.v1"
    )


def test_missing_required_lists_every_missing_field() -> None:
    findings = validate(_load("missing_required.yaml"))
    paths = {f.path for f in findings}
    for required in ("id", "name", "roles", "input_modalities", "skills"):
        assert f"capability_card.{required}" in paths, f"expected {required} in {paths}"
    assert "capability_card.embodiment.arm" in paths
    assert "capability_card.embodiment.dof" in paths
    assert "capability_card.embodiment.action_space" in paths


def test_missing_field_messages_carry_schema_url() -> None:
    findings = validate(_load("missing_required.yaml"))
    for f in findings:
        assert "Schema: https://farm.dev/schemas/capability_card.v1" in f.message
