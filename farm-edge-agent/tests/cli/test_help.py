from __future__ import annotations

from click.testing import CliRunner
from farm_edge_agent.cli.main import cli

LEAF_COMMANDS: list[list[str]] = [
    ["quickstart"],
    ["login"],
    ["start"],
    ["run"],
    ["export"],
    ["calibrate"],
    ["verify"],
    ["version"],
    ["config"],
    ["config", "init"],
    ["config", "doctor"],
    ["config", "show"],
    ["config", "set"],
    ["card"],
    ["card", "validate"],
    ["doctor"],
    ["doctor", "cameras"],
    ["doctor", "network"],
    ["doctor", "real-arm"],
]


def test_top_level_help() -> None:
    runner = CliRunner()
    for flag in ("--help", "-h"):
        result = runner.invoke(cli, [flag])
        assert result.exit_code == 0
        assert "Usage" in result.output


def test_every_subcommand_accepts_help() -> None:
    runner = CliRunner()
    for argv in LEAF_COMMANDS:
        result = runner.invoke(cli, [*argv, "--help"])
        assert result.exit_code == 0, f"--help failed for {argv}: {result.output}"
        assert "Usage" in result.output, f"no Usage line for {argv}"
