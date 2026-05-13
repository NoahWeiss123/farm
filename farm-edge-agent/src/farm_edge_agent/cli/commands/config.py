"""`farm config` subcommands: init, doctor, show, set."""

from __future__ import annotations

import sys
from typing import Any

import click
import yaml

from ...config import (
    ConfigError,
    Severity,
    check,
    load_config,
    read_template,
    resolve_config_path,
)

REDACTED = "<redacted>"


@click.group(name="config")
def config_group() -> None:
    """Manage ~/.farm/config.yaml."""


@config_group.command("init")
def init_cmd() -> None:
    """Scaffold ~/.farm/config.yaml from the bundled template."""
    path = resolve_config_path()
    if path.exists():
        click.echo(f"config already exists at {path}; refusing to overwrite", err=True)
        sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(read_template())
    click.echo(f"wrote config to {path}")


@config_group.command("doctor")
def doctor_cmd() -> None:
    """Validate the config; exit non-zero on critical findings."""
    path = resolve_config_path()
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        click.echo(f"[critical] {exc}", err=True)
        sys.exit(1)

    findings = check(cfg)
    if not findings:
        click.echo(f"config ok ({path})")
        return

    critical = 0
    for finding in findings:
        click.echo(f"[{finding.severity.value}] {finding.code.value} {finding.message}")
        click.echo(f"    fix: {finding.fix}")
        if finding.severity is Severity.CRITICAL:
            critical += 1

    if critical:
        sys.exit(1)


@config_group.command("show")
def show_cmd() -> None:
    """Print the effective config with secrets redacted."""
    cfg = load_config()
    data = cfg.model_dump(mode="json", exclude_none=False)
    if data.get("api_key"):
        data["api_key"] = REDACTED
    click.echo(yaml.safe_dump(data, sort_keys=False).rstrip())


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def set_cmd(key: str, value: str) -> None:
    """Set a dotted-path key, e.g. `farm config set camera.wrist.device /dev/video1`."""
    path = resolve_config_path()
    if not path.exists():
        click.echo(f"config not found at {path}; run 'farm config init' first", err=True)
        sys.exit(1)
    raw = yaml.safe_load(path.read_text()) or {}
    parts = key.split(".")
    cursor: dict[str, Any] = raw
    for part in parts[:-1]:
        next_node = cursor.get(part)
        if not isinstance(next_node, dict):
            next_node = {}
            cursor[part] = next_node
        cursor = next_node
    cursor[parts[-1]] = _coerce(value)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    click.echo(f"set {key} = {value}")


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
