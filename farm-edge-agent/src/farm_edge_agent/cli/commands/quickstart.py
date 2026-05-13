from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.command("quickstart", help="One-shot: sign in, generate key, write config, run a sim task.")
def quickstart() -> None:
    stub("quickstart")
