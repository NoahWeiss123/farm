from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.group("card", help="Capability card utilities.")
def card() -> None:
    pass


@card.command("validate", help="Validate a capability card against the JSON Schema.")
@click.argument("file", required=False, type=click.Path())
def validate(file: str | None) -> None:
    stub("card validate")
