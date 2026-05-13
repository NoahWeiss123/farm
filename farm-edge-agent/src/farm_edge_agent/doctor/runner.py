from __future__ import annotations

import sys
from typing import IO

from farm_edge_agent.doctor import cameras as _cameras
from farm_edge_agent.doctor import network as _network
from farm_edge_agent.doctor import real_arm as _real_arm


def run_all(
    out: IO[str] | None = None,
    stream_in: IO[str] | None = None,
    skip_real_arm: bool = False,
) -> None:
    stream = out if out is not None else sys.stdout
    sin = stream_in if stream_in is not None else sys.stdin

    stream.write("== doctor: cameras ==\n")
    _cameras.run(out=stream)

    stream.write("\n== doctor: network ==\n")
    _network.run(out=stream)

    if skip_real_arm:
        stream.write("\n== doctor: real-arm == (skipped: non-interactive)\n")
        return

    stream.write("\n== doctor: real-arm ==\n")
    _real_arm.run_real_arm(sin, stream)
