from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.command("calibrate", help="Interactive camera intrinsics + hand-eye calibration.")
def calibrate() -> None:
    stub("calibrate")
