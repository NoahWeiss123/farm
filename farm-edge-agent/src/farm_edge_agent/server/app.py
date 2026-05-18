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
GET  /v1/cameras/{name}.jpg     — live JPEG from a MuJoCo camera (exterior,
                                  wrist, topdown). Always serves the latest
                                  frame; UI polls or sets <img> refresh.
GET  /v1/cameras/{name}.depth.png — false-color depth render of the same
                                  camera (showing what π0.5 *could* see if
                                  it had depth — purely interpretability).
GET  /v1/inspect                — current plan + active node + last action +
                                  π0.5 observation summary. JSON snapshot.
GET  /v1/inspect/stream         — SSE feed of inspect events.

CORS is wide-open so the Next.js dev server on :3000 can reach us.
"""

from __future__ import annotations

import asyncio
import io
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


_VALID_CAMS = {"exterior", "wrist", "topdown"}


async def get_camera_jpeg(request: web.Request) -> web.Response:
    """Render one MuJoCo camera and return it as a JPEG.

    The render path runs synchronously on the asyncio thread; it's fast
    (<10 ms for a 480p frame on Mac M-series) and avoids the complexity
    of shipping render state to a worker. We compress at quality=82 — a
    sensible sweet spot between bandwidth and visual fidelity.
    """
    name = request.match_info["name"]
    if name not in _VALID_CAMS:
        return web.json_response({"error": f"unknown camera {name!r}"}, status=404)
    supervisor: RunSupervisor = request.app["supervisor"]
    try:
        from PIL import Image  # local import — heavy dep, only needed here
        img = supervisor.render_camera(name, width=480, height=360)
    except Exception as e:
        return web.json_response({"error": f"render failed: {e}"}, status=500)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=82)
    return web.Response(
        body=buf.getvalue(),
        content_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


async def get_camera_depth(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    if name not in _VALID_CAMS:
        return web.json_response({"error": f"unknown camera {name!r}"}, status=404)
    supervisor: RunSupervisor = request.app["supervisor"]
    try:
        import numpy as np
        from PIL import Image
        depth = supervisor.render_camera_depth(name, width=320, height=240)
        # Far values clipped; near=blue, far=red — gradient lets the eye
        # parse depth without needing a colorbar.
        d = np.clip(depth, 0.3, 1.6)
        norm = (d - d.min()) / max(1e-6, d.max() - d.min())
        # turbo-ish ramp via three channel polynomials
        r = (255 * (0.13 + 4.1 * norm - 4.5 * norm**2 + norm**3)).clip(0, 255).astype("uint8")
        g = (255 * (0.05 + 1.8 * norm - 0.8 * norm**2)).clip(0, 255).astype("uint8")
        b = (255 * (0.85 - 1.6 * norm + 0.7 * norm**2)).clip(0, 255).astype("uint8")
        rgb = np.stack([r, g, b], axis=-1)
    except Exception as e:
        return web.json_response({"error": f"depth render failed: {e}"}, status=500)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return web.Response(
        body=buf.getvalue(),
        content_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


async def get_inspect(request: web.Request) -> web.Response:
    supervisor: RunSupervisor = request.app["supervisor"]
    return web.json_response(supervisor.inspect())


async def stream_inspect(request: web.Request) -> web.StreamResponse:
    supervisor: RunSupervisor = request.app["supervisor"]
    bus: EventBus = request.app["bus"]
    resp = web.StreamResponse(headers=SSE_HEADERS)
    await resp.prepare(request)
    await resp.write(_sse_format({"type": "inspect", **supervisor.inspect()}))
    q = await bus.subscribe("inspect")
    try:
        while not request.transport.is_closing():
            try:
                event = await asyncio.wait_for(q.get(), timeout=2.0)
                await resp.write(_sse_format(event))
            except TimeoutError:
                # 2 s heartbeat doubles as a periodic re-publish of the
                # inspect snapshot, so the UI never goes stale.
                await resp.write(
                    _sse_format({"type": "inspect", **supervisor.inspect()})
                )
    finally:
        bus.unsubscribe("inspect", q)
    return resp


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
        web.get("/v1/cameras/{name}.jpg", get_camera_jpeg),
        web.get("/v1/cameras/{name}.depth.png", get_camera_depth),
        web.get("/v1/inspect", get_inspect),
        web.get("/v1/inspect/stream", stream_inspect),
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
