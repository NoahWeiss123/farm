from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.group("config", help="Manage the Edge Agent config file.")
def config() -> None:
    pass


@config.command("init", help="Scaffold ~/.farm/config.yaml from template.")
def init() -> None:
    stub("config init")


@config.command("doctor", help="Validate config, surface fixable errors.")
def doctor() -> None:
    stub("config doctor")


@config.command("show", help="Print current effective config (redacts secrets).")
def show() -> None:
    stub("config show")


@config.command("set", help="Mutate config from CLI.")
@click.argument("path", required=False)
@click.argument("value", required=False)
def set_(path: str | None, value: str | None) -> None:
    stub("config set")
