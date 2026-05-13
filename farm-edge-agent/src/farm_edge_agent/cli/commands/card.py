"""farm card subcommand: validate capability cards."""

import sys

import click
from farm_edge_agent.capability_cards.loader import parse_file
from farm_edge_agent.capability_cards.validator import validate


@click.group()
def card() -> None:
    """Capability card operations."""


@card.command("validate")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def validate_cmd(file: str) -> None:
    """Validate a capability card against the JSON Schema."""
    data = parse_file(file)
    findings = validate(data)
    if findings:
        for f in findings:
            click.echo(f"[FARM-E2001] {f.message}", err=True)
        sys.exit(1)
    click.echo("OK")
