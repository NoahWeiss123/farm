"""`farm serve` — boot the local edge daemon (HTTP + SSE + ROS-TCP bridge)."""

from __future__ import annotations

import sys
import webbrowser

import click


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="HTTP bind address.")
@click.option("--port", default=8787, show_default=True,
              help="HTTP port for the dashboard / API.")
@click.option("--ros-port", default=10000, show_default=True,
              help="TCP port for the Quest ROS-TCP-Endpoint bridge.")
@click.option("--backend", type=click.Choice(["sim", "xarm"]), default="sim",
              show_default=True, help="Robot backend.")
@click.option("--arm-ip", default=None, help="UF850 IP address (required when --backend=xarm).")
@click.option("--envelope/--no-envelope", default=True,
              help="Enforce workspace envelope on real hardware (default on; off "
                   "disables the safety cube — use only when you know the table).")
@click.option("--cameras/--no-cameras", default=True,
              help="Spawn RealSense camera subprocesses. Off when you don't need "
                   "the D435 feeds (e.g. during Quest teleop), since librealsense "
                   "is unstable with two D435s on macOS.")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the dashboard in the default browser on start.")
def serve(
    host: str,
    port: int,
    ros_port: int,
    backend: str,
    arm_ip: str | None,
    envelope: bool,
    cameras: bool,
    open_browser: bool,
) -> None:
    """Run the FARM edge daemon (sim or real arm + dashboard + ROS-TCP bridge)."""
    if backend == "xarm":
        if not arm_ip:
            click.echo("--arm-ip is required when --backend=xarm", err=True)
            sys.exit(1)
        from farm_edge_agent.backends import XArmBackend
        from farm_edge_agent.drivers.xarm import (
            DEFAULT_ENVELOPE_MAX,
            DEFAULT_ENVELOPE_MIN,
        )
        env = (DEFAULT_ENVELOPE_MIN, DEFAULT_ENVELOPE_MAX) if envelope else None
        chosen = XArmBackend(arm_ip, envelope=env, cameras=cameras)
        click.echo(
            f"farm serve: backend=xarm arm_ip={arm_ip} envelope={'on' if envelope else 'OFF'} "
            f"cameras={'on' if cameras else 'OFF'}"
        )
    else:
        from farm_edge_agent.backends import SimBackend
        chosen = SimBackend()
        click.echo("farm serve: backend=sim")

    if open_browser:
        url = f"http://{host}:{port}/"
        webbrowser.open(url, new=2)

    from farm_edge_agent.server import run as serve_run
    serve_run(host=host, port=port, ros_port=ros_port, backend=chosen)


def register(cli: click.Group) -> None:
    cli.add_command(serve)
