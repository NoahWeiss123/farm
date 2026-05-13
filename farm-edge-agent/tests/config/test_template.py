from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from farm_edge_agent.cli.commands.config import config
from farm_edge_agent.config import read_template


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FARM_CONFIG", raising=False)
    return tmp_path


def test_template_parses_as_yaml() -> None:
    parsed = yaml.safe_load(read_template())
    assert parsed["driver"] in ("xarm", "franka", "lerobot-mock")
    assert parsed["camera"]["wrist"]["device"]
    assert parsed["safety"]["velocity_cap_mps"]
    assert "${FARM_API_KEY}" in parsed["api_key"]


def test_init_writes_template_and_prints_path(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FARM_API_KEY", "sk-test")
    runner = CliRunner()
    result = runner.invoke(config, ["init"])
    assert result.exit_code == 0, result.output
    cfg = tmp_home / ".farm" / "config.yaml"
    assert cfg.exists()
    assert str(cfg) in result.output


def test_init_refuses_to_overwrite(tmp_home: Path) -> None:
    cfg = tmp_home / ".farm" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("api_key: existing\n")
    runner = CliRunner()
    result = runner.invoke(config, ["init"])
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output
    assert cfg.read_text() == "api_key: existing\n"


def test_show_redacts_api_key(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FARM_API_KEY", "sk-supersecret-shouldnt-leak")
    runner = CliRunner()
    runner.invoke(config, ["init"])
    result = runner.invoke(config, ["show"])
    assert result.exit_code == 0, result.output
    assert "<redacted>" in result.output
    assert "sk-supersecret-shouldnt-leak" not in result.output


def test_set_writes_dotted_path(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FARM_API_KEY", "sk-test")
    runner = CliRunner()
    runner.invoke(config, ["init"])
    result = runner.invoke(
        config, ["set", "camera.wrist.device", "/dev/video2"]
    )
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load((tmp_home / ".farm" / "config.yaml").read_text())
    assert raw["camera"]["wrist"]["device"] == "/dev/video2"


def test_set_creates_intermediate_dicts(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FARM_API_KEY", "sk-test")
    runner = CliRunner()
    runner.invoke(config, ["init"])
    result = runner.invoke(
        config, ["set", "telemetry.advanced.tracing", "true"]
    )
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load((tmp_home / ".farm" / "config.yaml").read_text())
    assert raw["telemetry"]["advanced"]["tracing"] is True


def test_set_coerces_numbers(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FARM_API_KEY", "sk-test")
    runner = CliRunner()
    runner.invoke(config, ["init"])
    runner.invoke(config, ["set", "safety.velocity_cap_mps", "0.5"])
    raw = yaml.safe_load((tmp_home / ".farm" / "config.yaml").read_text())
    assert raw["safety"]["velocity_cap_mps"] == 0.5


def test_set_refuses_when_no_config(tmp_home: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(config, ["set", "driver", "xarm"])
    assert result.exit_code != 0


def test_doctor_exits_nonzero_on_critical(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FARM_API_KEY", raising=False)
    cfg = tmp_home / ".farm" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "api_key: ''\n"
        "driver: lerobot-mock\n"
        "camera:\n"
        "  wrist:\n"
        "    device: /dev/video0\n"
    )
    runner = CliRunner()
    result = runner.invoke(config, ["doctor"])
    assert result.exit_code == 1
    assert "FARM-E1004" in result.output


def test_doctor_exit_zero_when_only_warnings(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    device = tmp_path / "video0"
    device.touch()
    monkeypatch.setenv("FARM_API_KEY", "sk-test")
    cfg = tmp_home / ".farm" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        f"api_key: ${{FARM_API_KEY}}\n"
        f"driver: lerobot-mock\n"
        f"camera:\n"
        f"  wrist:\n"
        f"    device: {device}\n"
    )
    runner = CliRunner()
    result = runner.invoke(config, ["doctor"])
    assert result.exit_code == 0, result.output


def test_doctor_reports_missing_env_var(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FARM_API_KEY", raising=False)
    runner = CliRunner()
    runner.invoke(config, ["init"])
    result = runner.invoke(config, ["doctor"])
    assert result.exit_code == 1
    assert "FARM_API_KEY" in result.output
