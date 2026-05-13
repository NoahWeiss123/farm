"""Validate capability card data against the JSON Schema.

The validator returns one :class:`Finding` per problem. Enum mismatches carry
a Levenshtein-nearest suggestion in the ``did_you_mean`` field so the CLI can
render the structured E2001 message from DESIGN.md.
"""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator

_SCHEMA_URL = "https://farm.dev/schemas/capability_card.v1"


@dataclass(frozen=True)
class Finding:
    path: str
    message: str
    value: Any = None
    did_you_mean: str | None = None


def _load_schema() -> dict[str, Any]:
    text = (
        resources.files("farm_shared.schemas")
        .joinpath("capability_card.v1.json")
        .read_text()
    )
    return json.loads(text)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(
                min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


def _nearest(value: str, options: list[str]) -> str | None:
    if not options:
        return None
    # Normalize by the longer string so a short candidate like "ee_velocity"
    # does not beat a long-but-prefix-matching candidate like
    # "ee_pose_delta_base_frame" purely on absolute distance.
    return min(
        options,
        key=lambda opt: _levenshtein(value, opt) / max(len(value), len(opt), 1),
    )


def _format_path(path: tuple[Any, ...]) -> str:
    parts: list[str] = ["capability_card"]
    for p in path:
        if isinstance(p, int):
            parts[-1] = f"{parts[-1]}[{p}]"
        else:
            parts.append(str(p))
    return ".".join(parts)


def _extract_missing(message: str) -> str:
    # jsonschema phrasing: "'id' is a required property"
    if "'" in message:
        return message.split("'")[1]
    return ""


def validate(data: dict[str, Any]) -> list[Finding]:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    findings: list[Finding] = []
    errors = sorted(
        validator.iter_errors(data),
        key=lambda e: (list(e.absolute_path), e.message),
    )
    for error in errors:
        path = _format_path(tuple(error.absolute_path))
        if error.validator == "enum":
            value = error.instance
            allowed = [str(a) for a in error.validator_value]
            suggestion = (
                _nearest(str(value), allowed) if isinstance(value, str) else None
            )
            tail = f" Did you mean '{suggestion}'?" if suggestion else ""
            findings.append(
                Finding(
                    path=path,
                    message=(
                        f"{path}: '{value}' not in allowed set."
                        f"{tail} Schema: {_SCHEMA_URL}"
                    ),
                    value=value,
                    did_you_mean=suggestion,
                )
            )
        elif error.validator == "required":
            missing = _extract_missing(error.message)
            full = f"{path}.{missing}" if missing else path
            findings.append(
                Finding(
                    path=full,
                    message=f"{full}: missing required field. Schema: {_SCHEMA_URL}",
                )
            )
        else:
            findings.append(
                Finding(
                    path=path,
                    message=f"{path}: {error.message}. Schema: {_SCHEMA_URL}",
                )
            )
    return findings
