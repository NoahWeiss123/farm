from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.command("login", help="Open browser, store API key in ~/.farm/credentials.")
def login() -> None:
    stub("login")
