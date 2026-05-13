from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.command("verify", help="Verify run record signature + lock match.")
@click.argument("run_id", required=False)
def verify(run_id: str | None) -> None:
    stub("verify")
