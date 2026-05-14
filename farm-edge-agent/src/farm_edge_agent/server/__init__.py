"""Local edge daemon — HTTP + SSE API the dashboard talks to.

Boots a SimDriver, accepts POST /v1/runs, drives the RunLoop in a worker
thread, and broadcasts events to any number of SSE subscribers. The shape
of these endpoints matches what the Cloudflare worker will expose later,
so the UI doesn't need to change when migrating.
"""

from farm_edge_agent.server.app import build_app, run

__all__ = ["build_app", "run"]
