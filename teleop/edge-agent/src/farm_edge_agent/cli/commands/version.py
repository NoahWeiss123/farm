from __future__ import annotations

import json as _json

import click

from farm_edge_agent import __version__ as AGENT_VERSION

SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = ("1.2.0",)


@click.command("version", help="Print agent version and supported protocol versions.")
@click.pass_context
def version(ctx: click.Context) -> None:
    payload = {
        "agent_version": AGENT_VERSION,
        "protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
    }
    if ctx.obj and ctx.obj.get("json"):
        click.echo(_json.dumps(payload))
        return
    click.echo(f"farm-edge-agent {AGENT_VERSION}")
    click.echo(f"protocol {', '.join(SUPPORTED_PROTOCOL_VERSIONS)}")
