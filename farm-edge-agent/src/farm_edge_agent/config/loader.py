"""Read, expand, and validate ~/.farm/config.yaml."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .schema import Config

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class ConfigError(Exception):
    """Base class for config-loading failures."""


class ConfigNotFoundError(ConfigError):
    """The config file does not exist at the resolved path."""


class EnvVarMissingError(ConfigError):
    """A `${ENV_VAR}` reference in the config has no value in the environment."""


def default_config_path() -> Path:
    return Path.home() / ".farm" / "config.yaml"


def resolve_config_path(path: Path | None = None) -> Path:
    """Resolve which config file to use.

    Order: explicit `path`, then `$FARM_CONFIG`, then `~/.farm/config.yaml`.
    """
    if path is not None:
        return path
    env = os.environ.get("FARM_CONFIG")
    if env:
        return Path(env)
    return default_config_path()


def _expand_env_vars(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise EnvVarMissingError(
                f"environment variable '{name}' is not set "
                f"(referenced in config). fix: export {name}=..."
            )
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: Path | None = None) -> Config:
    """Load and validate config from `path`, `$FARM_CONFIG`, or `~/.farm/config.yaml`.

    Expands `${ENV_VAR}` references in `api_key` only — secrets-shaped values
    are the one place env interpolation is expected.
    """
    resolved = resolve_config_path(path)
    if not resolved.exists():
        raise ConfigNotFoundError(f"config file not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text()) or {}
    api_key = raw.get("api_key")
    if isinstance(api_key, str):
        raw["api_key"] = _expand_env_vars(api_key)
    return Config(**raw)
