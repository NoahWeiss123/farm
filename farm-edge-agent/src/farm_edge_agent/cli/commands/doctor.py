from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.group("doctor", invoke_without_command=True,
             help="Run preflight checks (arm, camera, network, protocol).")
@click.pass_context
def doctor(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None:
        stub("doctor")


@doctor.command("cameras", help="List cameras, intrinsics, last calibration timestamp.")
def cameras() -> None:
    stub("doctor cameras")


@doctor.command("network", help="Probe DNS, WebSocket upgrade, RTT, TLS, MTU, throughput.")
def network() -> None:
    stub("doctor network")


@doctor.command("real-arm", help="Interactive real-arm setup walk-through.")
def real_arm() -> None:
    stub("doctor real-arm")
