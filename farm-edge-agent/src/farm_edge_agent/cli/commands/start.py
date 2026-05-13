from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.command("start", help="Long-running Edge Agent connection to the dispatcher.")
def start() -> None:
    stub("start")
