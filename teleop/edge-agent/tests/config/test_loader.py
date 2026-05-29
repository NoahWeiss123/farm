from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from farm_edge_agent.config.loader import (
    ConfigNotFoundError,
    EnvVarMissingError,
    default_config_path,
    load_config,
    resolve_config_path,
)


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FARM_CONFIG", raising=False)
    return tmp_path


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip())


def test_default_config_path_under_home(tmp_home: Path) -> None:
    assert default_config_path() == tmp_home / ".farm" / "config.yaml"


def test_resolve_prefers_explicit_path(tmp_home: Path, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.yaml"
    assert resolve_config_path(explicit) == explicit


def test_resolve_prefers_env_over_default(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "override.yaml"
    monkeypatch.setenv("FARM_CONFIG", str(override))
    assert resolve_config_path() == override


def test_load_basic_config(tmp_home: Path) -> None:
    _write(
        tmp_home / ".farm" / "config.yaml",
        """
        api_key: sk-test
        driver: lerobot-mock
        camera:
          wrist:
            device: /dev/video0
        """,
    )
    cfg = load_config()
    assert cfg.api_key == "sk-test"
    assert cfg.driver == "lerobot-mock"
    assert cfg.camera.wrist.device == "/dev/video0"
    assert cfg.camera.overhead is None
    assert cfg.telemetry.upload_frames is True


def test_env_var_expansion_in_api_key(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FARM_API_KEY", "sk-from-env")
    _write(
        tmp_home / ".farm" / "config.yaml",
        """
        api_key: ${FARM_API_KEY}
        driver: lerobot-mock
        camera:
          wrist:
            device: /dev/video0
        """,
    )
    cfg = load_config()
    assert cfg.api_key == "sk-from-env"


def test_env_var_missing_raises_clear_error(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FARM_API_KEY", raising=False)
    _write(
        tmp_home / ".farm" / "config.yaml",
        """
        api_key: ${FARM_API_KEY}
        driver: lerobot-mock
        camera:
          wrist:
            device: /dev/video0
        """,
    )
    with pytest.raises(EnvVarMissingError) as excinfo:
        load_config()
    assert "FARM_API_KEY" in str(excinfo.value)


def test_farm_config_env_var_used_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom" / "farm.yaml"
    _write(
        custom,
        """
        api_key: sk-custom
        driver: lerobot-mock
        camera:
          wrist:
            device: /dev/video0
        """,
    )
    monkeypatch.setenv("FARM_CONFIG", str(custom))
    cfg = load_config()
    assert cfg.api_key == "sk-custom"


def test_missing_config_raises(tmp_home: Path) -> None:
    with pytest.raises(ConfigNotFoundError):
        load_config()
