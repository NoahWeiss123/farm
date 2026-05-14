"""aiohttp app — the local edge daemon's HTTP/SSE surface.

Routes
------
GET  /healthz                   — liveness probe
GET  /v1/scene                  — current scene spec (props the UI should draw)
GET  /v1/world                  — current world snapshot (joints, tcp, props)
GET  /v1/world/stream           — SSE stream of world snapshots + joint_state
POST /v1/runs                   — submit a new run (body: {"task": "..."})
GET  /v1/runs                   — list runs (in-memory + on-disk)
GET  /v1/runs/{id}              — run status + buffered events
GET  /v1/runs/{id}/events       — SSE stream of run events (replay + live)
GET  /v1/runs:stream            — SSE stream of run-state changes for the
                                  dashboard's runs list

CORS is wide-open so the Next.js dev server on :3000 can reach us.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp_cors
from aiohttp import web

from farm_edge_agent.server.bus import EventBus
from farm_edge_agent.server.supervisor import RunSupervisor

log = logging.getLogger("farm.server")

SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_format(event: dict[str, Any], event_type: str | None = None) -> bytes:
    lines = []
    if event_type:
        lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(event, separators=(',', ':'))}")
    lines.append("")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


async def healthz(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def get_scene(request: web.Request) -> web.Response:
    supervisor: RunSupervisor = request.app["supervisor"]
    return web.json_response(supervisor.scene_spec())


async def get_world(request: web.Request) -> web.Response:
    supervisor: RunSupervisor = request.app["supervisor"]
    return web.json_response(supervisor.snapshot_world())


async def stream_world(request: web.Request) -> web.StreamResponse:
    bus: EventBus = request.app["bus"]
    supervisor: RunSupervisor = request.app["supervisor"]
    resp = web.StreamResponse(headers=SSE_HEADERS)
    await resp.prepare(request)
    # Initial snapshot so the client renders the arm immediately.
    await resp.write(_sse_format({"type": "world_snapshot", **supervisor.snapshot_world()}))
    q = await bus.subscribe("world")
    try:
        while not request.transport.is_closing():
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                await resp.write(_sse_format(event))
            except TimeoutError:
                # SSE heartbeat keeps the connection alive through proxies.
                await resp.write(b": ping\n\n")
    finally:
        bus.unsubscribe("world", q)
    return resp


async def post_run(request: web.Request) -> web.Response:
    body = await request.json()
    if not isinstance(body, dict) or "task" not in body:
        return web.json_response({"error": "body must include 'task'"}, status=400)
    task = str(body["task"]).strip()
    if not task:
        return web.json_response({"error": "task is empty"}, status=400)
    supervisor: RunSupervisor = request.app["supervisor"]
    status = supervisor.submit_run(task)
    return web.json_response(_status_dict(status), status=202)


async def list_runs(request: web.Request) -> web.Response:
    supervisor: RunSupervisor = request.app["supervisor"]
    runs = supervisor.list_runs()
    runs.sort(key=lambda r: r.submitted_at, reverse=True)
    return web.json_response({"runs": [_status_dict(r) for r in runs]})


async def get_run(request: web.Request) -> web.Response:
    rid = request.match_info["run_id"]
    supervisor: RunSupervisor = request.app["supervisor"]
    status = supervisor.get_run(rid)
    if status is None:
        # Try to reconstruct from disk
        events = supervisor.replay_run(rid)
        if not events:
            return web.json_response({"error": "not found"}, status=404)
        from farm_edge_agent.server.supervisor import RunStatus
        first = events[0]
        last = events[-1]
        task = ""
        outcome = None
        if first.get("type") == "run_started":
            task = first.get("data", {}).get("task", "")
        if last.get("type") == "run_completed":
            outcome = last.get("data", {}).get("outcome")
        return web.json_response(
            {
                "status": _status_dict(
                    RunStatus(
                        run_id=rid,
                        task=task,
                        state=outcome or "unknown",
                        submitted_at=first.get("ts", 0),
                        completed_at=last.get("ts", 0) if outcome else None,
                        outcome=outcome,
                    )
                ),
                "events": events,
            }
        )
    return web.json_response(
        {
            "status": _status_dict(status),
            "events": supervisor.replay_run(rid),
        }
    )


async def stream_run_events(request: web.Request) -> web.StreamResponse:
    rid = request.match_info["run_id"]
    supervisor: RunSupervisor = request.app["supervisor"]
    bus: EventBus = request.app["bus"]
    resp = web.StreamResponse(headers=SSE_HEADERS)
    await resp.prepare(request)
    # Replay history first
    for event in supervisor.replay_run(rid):
        await resp.write(_sse_format(event))
    q = await bus.subscribe(f"run:{rid}")
    try:
        while not request.transport.is_closing():
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                await resp.write(_sse_format(event))
                if event.get("type") == "run_completed":
                    break
            except TimeoutError:
                await resp.write(b": ping\n\n")
    finally:
        bus.unsubscribe(f"run:{rid}", q)
    return resp


async def stream_runs_list(request: web.Request) -> web.StreamResponse:
    supervisor: RunSupervisor = request.app["supervisor"]
    bus: EventBus = request.app["bus"]
    resp = web.StreamResponse(headers=SSE_HEADERS)
    await resp.prepare(request)
    # Initial snapshot
    await resp.write(
        _sse_format(
            {
                "type": "runs_snapshot",
                "runs": [_status_dict(r) for r in supervisor.list_runs()],
            }
        )
    )
    q = await bus.subscribe("runs")
    try:
        while not request.transport.is_closing():
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                await resp.write(_sse_format(event))
            except TimeoutError:
                await resp.write(b": ping\n\n")
    finally:
        bus.unsubscribe("runs", q)
    return resp


def _status_dict(status: object) -> dict[str, Any]:
    d = dict(status.__dict__)
    return d


def build_app() -> web.Application:
    app = web.Application(client_max_size=2_000_000)
    bus = EventBus(history=400)
    supervisor = RunSupervisor(bus)
    app["bus"] = bus
    app["supervisor"] = supervisor

    async def _on_startup(app: web.Application) -> None:
        bus.attach_loop(asyncio.get_running_loop())

    app.on_startup.append(_on_startup)

    routes = [
        web.get("/healthz", healthz),
        web.get("/v1/scene", get_scene),
        web.get("/v1/world", get_world),
        web.get("/v1/world/stream", stream_world),
        web.post("/v1/runs", post_run),
        web.get("/v1/runs", list_runs),
        web.get("/v1/runs:stream", stream_runs_list),
        web.get("/v1/runs/{run_id}", get_run),
        web.get("/v1/runs/{run_id}/events", stream_run_events),
    ]
    for route in routes:
        app.router.add_route(route.method, route.path, route.handler)

    # CORS for the Next.js dev server.
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=False,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
            )
        },
    )
    for r in list(app.router.routes()):
        cors.add(r)

    return app


def run(host: str = "127.0.0.1", port: int = 8787) -> None:
    logging.basicConfig(level=logging.INFO)
    app = build_app()
    log.info("farm edge daemon listening on http://%s:%d", host, port)
    web.run_app(app, host=host, port=port, print=None)


__all__ = ["build_app", "run"]
