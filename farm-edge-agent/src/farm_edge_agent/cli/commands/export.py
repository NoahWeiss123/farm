from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.command("export", help="Download run record as JSONL + LeRobot shards.")
@click.argument("run_id", required=False)
def export(run_id: str | None) -> None:
    stub("export")
