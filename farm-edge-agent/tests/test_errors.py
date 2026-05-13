from __future__ import annotations

import json
import re
import string
from pathlib import Path

import pytest
from farm_edge_agent.errors import FarmError, Severity, emit_to_cli
from farm_shared.errors import ErrorCode

_DOCS_PATH = Path(__file__).resolve().parents[2] / "docs" / "errors.md"


def _stub_slots(code: ErrorCode) -> dict[str, str]:
    names = {n for _, n, _, _ in string.Formatter().parse(code.template) if n}
    return {n: f"<{n}>" for n in names}


def test_str_matches_format_error_for_e1001() -> None:
    err = FarmError(ErrorCode.E1001, device="/dev/video0")
    assert str(err) == (
        "[FARM-E1001] No camera found at /dev/video0 — fix: "
        "'farm doctor cameras', then 'farm config set camera.wrist.device /dev/videoN'"
    )


def test_emit_text_prints_exact_expected_string(capsys: pytest.CaptureFixture[str]) -> None:
    err = FarmError(ErrorCode.E1001, device="/dev/video0")
    with pytest.raises(SystemExit):
        emit_to_cli(err, json=False)
    captured = capsys.readouterr()
    assert captured.err == (
        "[FARM-E1001] No camera found at /dev/video0 — fix: "
        "'farm doctor cameras', then 'farm config set camera.wrist.device /dev/videoN'\n"
    )
    assert captured.out == ""


def test_emit_json_is_parseable_and_has_required_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    err = FarmError(ErrorCode.E1001, device="/dev/video0")
    with pytest.raises(SystemExit):
        emit_to_cli(err, json=True)
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert set(payload) == {"code", "message", "fix", "docs_url"}
    assert payload["code"] == "FARM-E1001"
    assert payload["message"] == "No camera found at /dev/video0"
    assert payload["fix"] == (
        "'farm doctor cameras', then 'farm config set camera.wrist.device /dev/videoN'"
    )
    assert payload["docs_url"] == "https://farm.dev/errors/E1001"


def test_emit_json_fix_is_null_when_template_has_no_fix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    err = FarmError(ErrorCode.E3001)
    with pytest.raises(SystemExit):
        emit_to_cli(err, json=True)
    payload = json.loads(capsys.readouterr().err)
    assert payload["fix"] is None
    assert payload["message"].startswith("Safety envelope violation")


def test_configuration_severity_exits_with_2() -> None:
    err = FarmError(ErrorCode.E1001, device="/dev/video0")
    assert err.severity is Severity.CONFIGURATION
    assert err.exit_code == 2
    with pytest.raises(SystemExit) as exc:
        emit_to_cli(err, json=False)
    assert exc.value.code == 2


def test_runtime_severity_exits_with_1() -> None:
    err = FarmError(ErrorCode.E3001)
    assert err.severity is Severity.RUNTIME
    assert err.exit_code == 1
    with pytest.raises(SystemExit) as exc:
        emit_to_cli(err, json=False)
    assert exc.value.code == 1


def test_every_error_code_has_a_severity_assignment() -> None:
    for code in ErrorCode:
        err = FarmError(code, **_stub_slots(code))
        assert err.severity in (Severity.CONFIGURATION, Severity.RUNTIME)
        assert err.exit_code in (1, 2)


def test_docs_url_matches_design_link_template() -> None:
    for code in ErrorCode:
        err = FarmError(code, **_stub_slots(code))
        assert err.docs_url == f"https://farm.dev/errors/{code.name}"


def test_every_error_code_has_a_docs_section() -> None:
    text = _DOCS_PATH.read_text(encoding="utf-8")
    headings = set(re.findall(r"^##\s+(FARM-E\d{4})\s*$", text, flags=re.MULTILINE))
    expected = {f"FARM-{code.name}" for code in ErrorCode}
    missing = expected - headings
    assert not missing, f"missing sections in docs/errors.md: {sorted(missing)}"


def test_docs_links_to_canonical_url_per_code() -> None:
    text = _DOCS_PATH.read_text(encoding="utf-8")
    for code in ErrorCode:
        assert f"https://farm.dev/errors/{code.name}" in text, (
            f"docs/errors.md missing link for {code.name}"
        )


def test_to_dict_round_trips_through_json() -> None:
    err = FarmError(ErrorCode.E1006, agent_version="1.0", required_version="1.2")
    payload = json.loads(json.dumps(err.to_dict()))
    assert payload["code"] == "FARM-E1006"
    assert "Edge Agent v1.0" in payload["message"]
    assert payload["fix"] == "'pip install -U farm-edge-agent'"
    assert payload["docs_url"] == "https://farm.dev/errors/E1006"
