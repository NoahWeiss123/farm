from __future__ import annotations

import json
import re

from click.testing import CliRunner

from farm_edge_agent import __version__ as AGENT_VERSION
from farm_edge_agent.cli.main import cli
from farm_edge_agent.cli.commands.version import SUPPORTED_PROTOCOL_VERSIONS

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

SUBCOMMANDS = [
    "quickstart",
    "login",
    "config",
    "start",
    "run",
    "export",
    "calibrate",
    "card",
    "doctor",
    "verify",
    "version",
]


def test_help_lists_every_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.output, f"missing subcommand in help: {name}"


def test_no_args_prints_help_and_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.output


def test_unknown_subcommand_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["definitely-not-a-command"])
    assert result.exit_code != 0
    assert "No such command" in result.output or "Usage" in result.output


def test_version_prints_agent_and_protocol() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert AGENT_VERSION in result.output
    assert SEMVER_RE.match(AGENT_VERSION), f"agent version not semver: {AGENT_VERSION}"
    for proto in SUPPORTED_PROTOCOL_VERSIONS:
        assert proto in result.output
        assert SEMVER_RE.match(proto), f"protocol version not semver: {proto}"


def test_version_json_emits_structured_payload() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--json", "version"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["agent_version"] == AGENT_VERSION
    assert payload["protocol_versions"] == list(SUPPORTED_PROTOCOL_VERSIONS)


def test_global_flags_accepted() -> None:
    runner = CliRunner()
    args = [
        "--config", "/tmp/farm.yaml",
        "--workspace", "demo",
        "--quiet",
        "--auto-update",
        "--accept-calibration",
        "version",
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0


def test_stub_subcommands_exit_zero() -> None:
    runner = CliRunner()
    stubs = [
        ["quickstart"],
        ["login"],
        ["start"],
        ["run", "pick the red block"],
        ["run", "--offline", "task"],
        ["run", "--resume", "r_123"],
        ["export", "r_123"],
        ["calibrate"],
        ["verify", "r_123"],
        ["config", "init"],
        ["config", "doctor"],
        ["config", "show"],
        ["config", "set", "camera.wrist.device", "/dev/video0"],
        ["card", "validate", "card.yaml"],
        ["doctor"],
        ["doctor", "cameras"],
        ["doctor", "network"],
        ["doctor", "real-arm"],
    ]
    for argv in stubs:
        result = runner.invoke(cli, argv)
        assert result.exit_code == 0, f"failed: {argv} -> {result.output}"
        assert "[FARM] not implemented yet" in result.output, f"missing stub marker: {argv}"
