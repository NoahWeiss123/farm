"""Parse and validate capability card files."""

import json
from pathlib import Path
from typing import Any

import yaml
from farm_shared.capability_card import CapabilityCard
from farm_shared.errors import ErrorCode

from farm_edge_agent.capability_cards.validator import validate
from farm_edge_agent.errors import FarmError


def parse_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if p.suffix == ".json":
        return json.loads(text)
    raise FarmError(
        ErrorCode.E2001,
        value=p.suffix or "<no extension>",
        suggestion=".yaml",
        did_you_mean=".yaml",
    )


def load_card(path: str | Path) -> CapabilityCard:
    data = parse_file(path)
    findings = validate(data)
    if findings:
        first = findings[0]
        raise FarmError(
            ErrorCode.E2001,
            value="" if first.value is None else first.value,
            suggestion=first.did_you_mean or "",
            did_you_mean=first.did_you_mean or "",
        )
    return CapabilityCard.model_validate(data)
