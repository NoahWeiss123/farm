"""`farm serve` — boot the local edge daemon (HTTP + SSE)."""

from __future__ import annotations

import click


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="HTTP bind address.")
@click.option("--port", default=8787, show_default=True, help="HTTP port.")
def serve(host: str, port: int) -> None:
    """Run the FARM edge daemon that the dashboard talks to."""
    # Defer the heavy import (mujoco compile) until the command actually runs.
    from farm_edge_agent.server import run as serve_run

    serve_run(host=host, port=port)


def register(cli: click.Group) -> None:
    cli.add_command(serve)
