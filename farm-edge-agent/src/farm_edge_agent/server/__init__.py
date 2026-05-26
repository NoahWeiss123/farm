"""Local edge daemon — HTTP + SSE API the FARM dashboard talks to.

Hosts: the sim, the webviz-style dashboard (served at /), jog/home/gripper
control endpoints, live camera renders, and the SSE world stream. The
ROS-TCP-Endpoint wire bridge runs alongside in its own listener thread.
"""

from farm_edge_agent.server.app import build_app, run

__all__ = ["build_app", "run"]
