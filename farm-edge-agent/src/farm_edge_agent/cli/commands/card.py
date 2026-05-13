"""farm card subcommand: validate capability cards."""

from __future__ import annotations

import sys

import click

from farm_edge_agent.capability_cards.loader import parse_file
from farm_edge_agent.capability_cards.validator import validate


@click.group(name="card", help="Capability card utilities.")
def card() -> None:
    """Capability card operations."""


@card.command("validate", help="Validate a capability card against the JSON Schema.")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def validate_cmd(file: str) -> None:
    data = parse_file(file)
    findings = validate(data)
    if findings:
        for f in findings:
            click.echo(f"[FARM-E2001] {f.message}", err=True)
        sys.exit(1)
    click.echo("OK")
