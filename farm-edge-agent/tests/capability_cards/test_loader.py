"""Tests for capability_cards.loader."""

import json
from pathlib import Path

import pytest
from farm_edge_agent.capability_cards.loader import load_card, parse_file
from farm_edge_agent.errors import FarmError

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_valid_yaml_round_trips() -> None:
    card = load_card(FIXTURES / "valid_pi05.yaml")
    assert card.id == "pi05-ufactory-ft-v1"


def test_parse_yml_extension(tmp_path: Path) -> None:
    src = FIXTURES / "valid_pi05.yaml"
    dst = tmp_path / "card.yml"
    dst.write_text(src.read_text())
    card = load_card(dst)
    assert card.id == "pi05-ufactory-ft-v1"


def test_parse_json_extension(tmp_path: Path) -> None:
    data = parse_file(FIXTURES / "valid_pi05.yaml")
    dst = tmp_path / "card.json"
    dst.write_text(json.dumps(data))
    card = load_card(dst)
    assert card.id == "pi05-ufactory-ft-v1"


def test_invalid_action_space_raises_farm_error_with_suggestion() -> None:
    with pytest.raises(FarmError) as exc_info:
        load_card(FIXTURES / "invalid_action_space.yaml")
    err = exc_info.value
    assert getattr(err, "did_you_mean", None) == "ee_pose_delta_base_frame"


def test_missing_required_raises_farm_error() -> None:
    with pytest.raises(FarmError):
        load_card(FIXTURES / "missing_required.yaml")


def test_unknown_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "card.txt"
    bad.write_text("not a card")
    with pytest.raises(FarmError):
        parse_file(bad)
