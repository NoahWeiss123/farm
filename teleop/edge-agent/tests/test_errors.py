from __future__ import annotations

import json
import string

import pytest
from farm_edge_agent.errors import FarmError, Severity, emit_to_cli
from farm_shared.errors import ErrorCode


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


def test_to_dict_round_trips_through_json() -> None:
    err = FarmError(ErrorCode.E1006, agent_version="1.0", required_version="1.2")
    payload = json.loads(json.dumps(err.to_dict()))
    assert payload["code"] == "FARM-E1006"
    assert "Edge Agent v1.0" in payload["message"]
    assert payload["fix"] == "'pip install -U farm-edge-agent'"
