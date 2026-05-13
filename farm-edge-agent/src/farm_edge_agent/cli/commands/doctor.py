from __future__ import annotations

import sys

import click

from farm_edge_agent.doctor import cameras as _cameras
from farm_edge_agent.doctor import network as _network
from farm_edge_agent.doctor import real_arm as _real_arm
from farm_edge_agent.doctor import runner as _runner


@click.group(
    "doctor",
    invoke_without_command=True,
    help="Run preflight checks (cameras, network, real-arm).",
)
@click.pass_context
def doctor(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    skip = not sys.stdin.isatty()
    _runner.run_all(out=sys.stdout, stream_in=sys.stdin, skip_real_arm=skip)


@doctor.command(
    "cameras", help="List cameras, intrinsics, last calibration timestamp."
)
def cameras() -> None:
    _cameras.run(out=sys.stdout)


@doctor.command(
    "network",
    help="Probe DNS, WebSocket upgrade, RTT, TLS, MTU, throughput.",
)
def network() -> None:
    final, _findings = _network.run(out=sys.stdout)
    if final is _network.Status.FAILED:
        sys.exit(1)


@doctor.command("real-arm", help="Interactive real-arm setup walk-through.")
def real_arm() -> None:
    _real_arm.run_real_arm(sys.stdin, sys.stdout)
