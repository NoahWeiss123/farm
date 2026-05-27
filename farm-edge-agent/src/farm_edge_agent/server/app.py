"""aiohttp app — local edge daemon serving the dashboard + control API.

Routes
------
GET  /                          — webviz-style dashboard (static)
GET  /healthz                   — liveness probe
GET  /v1/world                  — current backend snapshot (joints, TCP, gripper)
GET  /v1/world/stream           — SSE stream of world snapshots
POST /v1/teleop/jog             — {axis, sign, step_mm?, step_rad?}
POST /v1/teleop/home            — drive arm to backend home pose
POST /v1/teleop/gripper         — {state: "open"|"closed"}
POST /v1/teleop/estop           — software emergency stop
POST /v1/teleop/estop/clear     — re-arm after e-stop
GET  /v1/teleop/filter          — current 1€ joint-filter params
POST /v1/teleop/filter          — {min_cutoff?, beta?} hot-tune 1€ filter
GET  /v1/cameras/{name}.jpg     — live JPEG from a backend camera (placeholder for xarm)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any

import aiohttp_cors
from aiohttp import web

from farm_edge_agent.backends.base import RobotBackend
from farm_edge_agent.server.bus import EventBus
from farm_edge_agent.server.supervisor import Supervisor

log = logging.getLogger("farm.server")

SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
# parents: server → farm_edge_agent → src → farm-edge-agent → CS153 repo
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
DATASETS_DIR = Path(__file__).resolve().parents[4] / "datasets"
_VALID_AXES = {"x", "y", "z", "rx", "ry", "rz"}
DEFAULT_STEP_MM = 5.0
DEFAULT_STEP_RAD = math.radians(2.0)

# Episode IDs look like ``episode_<UTC>_<uuid8>``; this also covers any
# legacy directories that happen to be in datasets/. The regex is the
# only thing standing between a user-supplied path segment and shutil's
# rmtree, so keep it tight.
_EPISODE_ID_RE = re.compile(r"^episode_[A-Za-z0-9_\-]+$")
_CAM_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _episode_dir(name: str) -> Path | None:
    if not _EPISODE_ID_RE.match(name or ""):
        return None
    p = (DATASETS_DIR / name).resolve()
    try:
        p.relative_to(DATASETS_DIR.resolve())
    except ValueError:
        return None
    if not p.is_dir():
        return None
    return p


def _sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


# ── routes ──────────────────────────────────────────────────────────────────


async def healthz(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def get_world(request: web.Request) -> web.Response:
    # Off the event loop: snapshot() takes the backend lock, which is
    # also held during camera renders / IK / GL init. A synchronous call
    # here would freeze the entire aiohttp loop (even /healthz) the
    # moment the lock is contended.
    snap = await asyncio.to_thread(request.app["supervisor"].snapshot)
    return web.json_response(snap)


async def stream_world(request: web.Request) -> web.StreamResponse:
    bus: EventBus = request.app["bus"]
    supervisor: Supervisor = request.app["supervisor"]
    resp = web.StreamResponse(headers=SSE_HEADERS)
    await resp.prepare(request)
    await resp.write(_sse({"type": "world_snapshot", **supervisor.snapshot()}))
    q = await bus.subscribe("world")
    try:
        while not request.transport.is_closing():
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                await resp.write(_sse(event))
            except TimeoutError:
                await resp.write(b": ping\n\n")
    finally:
        bus.unsubscribe("world", q)
    return resp


async def post_jog(request: web.Request) -> web.Response:
    body = await request.json()
    if not isinstance(body, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    axis = str(body.get("axis", "")).lower()
    if axis not in _VALID_AXES:
        return web.json_response(
            {"error": f"axis must be one of {sorted(_VALID_AXES)}; got {axis!r}"},
            status=400,
        )
    sign = body.get("sign")
    if sign not in (-1, 1):
        return web.json_response({"error": "sign must be -1 or +1"}, status=400)
    step_mm = float(body.get("step_mm", DEFAULT_STEP_MM))
    step_rad = float(body.get("step_rad", DEFAULT_STEP_RAD))
    supervisor: Supervisor = request.app["supervisor"]
    try:
        result = await asyncio.to_thread(
            supervisor.jog, axis, sign, step_mm=step_mm, step_rad=step_rad
        )
    except Exception as exc:
        log.warning("jog rejected: %s", exc)
        return web.json_response({"error": str(exc)}, status=409)
    return web.json_response(result)


async def post_home(request: web.Request) -> web.Response:
    supervisor: Supervisor = request.app["supervisor"]
    try:
        snap = await asyncio.to_thread(supervisor.home)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=409)
    return web.json_response(snap)


async def post_gripper(request: web.Request) -> web.Response:
    body = await request.json()
    if not isinstance(body, dict) or "state" not in body:
        return web.json_response({"error": "body must include 'state'"}, status=400)
    state = str(body["state"]).lower()
    if state not in ("open", "closed"):
        return web.json_response({"error": "state must be 'open' or 'closed'"}, status=400)
    supervisor: Supervisor = request.app["supervisor"]
    try:
        snap = await asyncio.to_thread(supervisor.set_gripper, state)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=409)
    return web.json_response(snap)


async def post_estop(request: web.Request) -> web.Response:
    supervisor: Supervisor = request.app["supervisor"]
    result = await asyncio.to_thread(supervisor.estop)
    return web.json_response(result)


async def post_estop_clear(request: web.Request) -> web.Response:
    supervisor: Supervisor = request.app["supervisor"]
    result = await asyncio.to_thread(supervisor.estop_clear)
    return web.json_response(result)


async def post_ghost_pose(request: web.Request) -> web.Response:
    """Test hook for the Quest bridge — POST a target TCP pose and watch
    the dashboard's ghost arm follow it. Body: ``{"pose": [x,y,z,rx,ry,rz]}``
    in millimetres + degrees."""
    body = await request.json()
    pose = body.get("pose")
    if not (isinstance(pose, list) and len(pose) == 6):
        return web.json_response(
            {"error": "body must be {pose: [x,y,z,rx,ry,rz] in mm+deg}"},
            status=400,
        )
    supervisor: Supervisor = request.app["supervisor"]
    result = await asyncio.to_thread(supervisor.set_ghost_target_pose, tuple(pose))
    return web.json_response(result)


async def post_cameras_swap(request: web.Request) -> web.Response:
    supervisor: Supervisor = request.app["supervisor"]
    result = await asyncio.to_thread(supervisor.swap_cameras)
    return web.json_response(result)


async def get_camera_jpeg(request: web.Request) -> web.Response:
    """Pass-through real camera JPEGs ONLY.

    Camera tiles in the dashboard represent physical hardware. There is
    no sim render fallback — when the backend has no live camera (sim,
    or xarm with disconnected RealSenses), we return 503 and the
    dashboard paints a black tile.
    """
    name = request.match_info["name"]
    supervisor: Supervisor = request.app["supervisor"]
    backend = getattr(supervisor, "_backend", None)
    fast = getattr(backend, "camera_jpeg", None)
    if not callable(fast):
        return web.Response(status=503, headers={"Cache-Control": "no-store"})
    try:
        blob = await asyncio.to_thread(fast, name)
    except Exception as exc:
        log.debug("camera fetch failed: %s", exc)
        return web.Response(status=503, headers={"Cache-Control": "no-store"})
    if blob is None:
        return web.Response(status=503, headers={"Cache-Control": "no-store"})
    return web.Response(
        body=blob, content_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


async def post_drive_mode(request: web.Request) -> web.Response:
    """Toggle / set the right-trigger drive mode.

    Body: ``{"drive_real_arm": true|false}`` to set explicitly, or
    empty body / ``{}`` to flip the current value. Mirrors the
    right-stick-click toggle on the Quest controller so the dashboard
    can do the same thing.
    """
    supervisor: Supervisor = request.app["supervisor"]
    backend = getattr(supervisor, "_backend", None)
    if backend is None or not hasattr(backend, "drive_real_arm"):
        return web.json_response({"error": "backend has no drive_real_arm"}, status=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if isinstance(body, dict) and "drive_real_arm" in body:
        target = bool(body["drive_real_arm"])
    else:
        target = not bool(backend.drive_real_arm)
    # Mode switching may touch the SDK — run off the event loop.
    await asyncio.to_thread(lambda: setattr(backend, "drive_real_arm", target))
    return web.json_response({"drive_real_arm": bool(backend.drive_real_arm)})


async def get_filter_params(request: web.Request) -> web.Response:
    """Current 1€ filter params on the joint stream."""
    supervisor: Supervisor = request.app["supervisor"]
    backend = getattr(supervisor, "_backend", None)
    if backend is None or not hasattr(backend, "get_joint_filter_params"):
        return web.json_response({"error": "backend has no joint filter"}, status=503)
    return web.json_response(backend.get_joint_filter_params())


async def post_filter_params(request: web.Request) -> web.Response:
    """Hot-tune the 1€ filter from the dashboard.

    Body: ``{"min_cutoff": <Hz>, "beta": <float>}`` — either key may
    be omitted to leave that channel untouched. Echoes the new params
    back so the UI can stay authoritative.
    """
    supervisor: Supervisor = request.app["supervisor"]
    backend = getattr(supervisor, "_backend", None)
    if backend is None or not hasattr(backend, "set_joint_filter_params"):
        return web.json_response({"error": "backend has no joint filter"}, status=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    kwargs: dict[str, float] = {}
    for key in ("min_cutoff", "beta"):
        if key in body and body[key] is not None:
            try:
                kwargs[key] = float(body[key])
            except (TypeError, ValueError):
                return web.json_response(
                    {"error": f"{key} must be a number"}, status=400
                )
    params = await asyncio.to_thread(
        lambda: backend.set_joint_filter_params(**kwargs)
    )
    return web.json_response(params)


async def post_recording_start(request: web.Request) -> web.Response:
    rec = request.app["supervisor"].recorder
    if rec is None:
        return web.json_response({"error": "recorder not configured"}, status=503)
    return web.json_response(rec.start())


async def post_recording_save(request: web.Request) -> web.Response:
    rec = request.app["supervisor"].recorder
    if rec is None:
        return web.json_response({"error": "recorder not configured"}, status=503)
    return web.json_response(rec.stop_save())


async def post_recording_cancel(request: web.Request) -> web.Response:
    rec = request.app["supervisor"].recorder
    if rec is None:
        return web.json_response({"error": "recorder not configured"}, status=503)
    return web.json_response(rec.cancel())


async def get_recording_state(request: web.Request) -> web.Response:
    rec = request.app["supervisor"].recorder
    if rec is None:
        return web.json_response({"recording": False, "error": "not configured"})
    return web.json_response(rec.state)


def _hud_state_sync(supervisor: Supervisor) -> dict[str, Any]:
    backend = supervisor.backend
    grabber = getattr(backend, "_grabber", None)
    alive_fn = getattr(grabber, "alive", None) if grabber is not None else None
    if callable(alive_fn):
        cam_alive = alive_fn()
    else:
        cam_alive = {name: False for name in supervisor.cameras()}
    cameras = [{"name": n, "alive": bool(v)} for n, v in cam_alive.items()]

    ep_count = 0
    if DATASETS_DIR.is_dir():
        for p in DATASETS_DIR.iterdir():
            if p.is_dir() and _EPISODE_ID_RE.match(p.name):
                ep_count += 1

    rec = supervisor.recorder
    rec_state = rec.state if rec is not None else {"recording": False}

    return {
        "cameras": cameras,
        "episodes": ep_count,
        "recording": rec_state,
        "drive_real_arm": bool(getattr(backend, "drive_real_arm", False)),
    }


async def get_hud(request: web.Request) -> web.Response:
    supervisor: Supervisor = request.app["supervisor"]
    payload = await asyncio.to_thread(_hud_state_sync, supervisor)
    return web.json_response(payload)


def _list_episodes_sync() -> list[dict[str, Any]]:
    if not DATASETS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in DATASETS_DIR.iterdir():
        if not p.is_dir() or not _EPISODE_ID_RE.match(p.name):
            continue
        meta_path = p / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        cameras = meta.get("cameras") or []
        if not cameras:
            cam_root = p / "cameras"
            if cam_root.is_dir():
                cameras = sorted(c.name for c in cam_root.iterdir() if c.is_dir())
        out.append({
            "id": p.name,
            "start_iso": meta.get("start_iso"),
            "duration_s": meta.get("duration_s"),
            "fps": meta.get("fps"),
            "frame_count": meta.get("frame_count"),
            "cameras": list(cameras),
            "backend": meta.get("backend"),
        })
    # Newest first: episode IDs embed the UTC timestamp so a lexicographic
    # sort matches chronological order without parsing meta.json.
    out.sort(key=lambda e: e["id"], reverse=True)
    return out


async def list_episodes(_: web.Request) -> web.Response:
    episodes = await asyncio.to_thread(_list_episodes_sync)
    return web.json_response({"episodes": episodes})


async def get_episode_meta(request: web.Request) -> web.Response:
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.json_response({"error": "not found"}, status=404)
    meta = p / "meta.json"
    if not meta.is_file():
        return web.json_response({"error": "no meta"}, status=404)
    return web.Response(body=meta.read_bytes(), content_type="application/json")


async def get_episode_frames(request: web.Request) -> web.Response:
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.json_response({"error": "not found"}, status=404)
    frames = p / "frames.jsonl"
    if not frames.is_file():
        return web.Response(text="", content_type="application/x-ndjson")
    return web.Response(body=frames.read_bytes(), content_type="application/x-ndjson")


def _episode_frame_path(episode: Path, cam: str, idx: int) -> Path | None:
    if not _CAM_NAME_RE.match(cam or "") or idx < 0:
        return None
    candidate = (episode / "cameras" / cam / f"{idx:06d}.jpg").resolve()
    try:
        candidate.relative_to(episode.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


async def get_episode_thumbnail(request: web.Request) -> web.Response:
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.Response(status=404)
    cam_root = p / "cameras"
    if not cam_root.is_dir():
        return web.Response(status=404)
    # Prefer "base" — that's the wide context shot — then fall back to
    # whichever camera directory has frame 0 on disk.
    cam_order: list[str] = []
    if (cam_root / "base").is_dir():
        cam_order.append("base")
    for c in sorted(cam_root.iterdir()):
        if c.is_dir() and c.name not in cam_order:
            cam_order.append(c.name)
    for cam in cam_order:
        frame = _episode_frame_path(p, cam, 0)
        if frame is not None:
            return web.Response(
                body=frame.read_bytes(), content_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
    return web.Response(status=404)


async def get_episode_camera_frame(request: web.Request) -> web.Response:
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.Response(status=404)
    cam = request.match_info["cam"]
    try:
        idx = int(request.match_info["idx"])
    except ValueError:
        return web.Response(status=400)
    frame = _episode_frame_path(p, cam, idx)
    if frame is None:
        return web.Response(status=404)
    return web.Response(
        body=frame.read_bytes(), content_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


async def delete_episode(request: web.Request) -> web.Response:
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.json_response({"error": "not found"}, status=404)
    rec = request.app["supervisor"].recorder
    if rec is not None:
        st = rec.state
        if st.get("recording") and st.get("episode_id") == p.name:
            return web.json_response(
                {"error": "cannot delete an episode that is still recording"},
                status=409,
            )
    try:
        await asyncio.to_thread(shutil.rmtree, p)
    except Exception as exc:
        log.warning("episode delete failed for %s: %s", p, exc)
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"ok": True, "id": p.name})


async def serve_dashboard(_: web.Request) -> web.Response:
    index = WEB_DIR / "index.html"
    if not index.is_file():
        return web.Response(text="dashboard not built (missing web/index.html)", status=500)
    return web.Response(body=index.read_bytes(), content_type="text/html")


# ── wiring ──────────────────────────────────────────────────────────────────


def build_app(*, backend: RobotBackend | None = None) -> web.Application:
    """Build the aiohttp app. If ``backend`` is omitted, the sim backend
    is used (so existing tests keep working). Production callers pass
    a configured backend (SimBackend or XArmBackend) explicitly."""
    if backend is None:
        from farm_edge_agent.backends import SimBackend
        backend = SimBackend()

    app = web.Application(client_max_size=2_000_000)
    bus = EventBus(history=400)
    supervisor = Supervisor(bus, backend=backend)
    from farm_edge_agent.recorder import Recorder

    recorder = Recorder(supervisor, datasets_dir=DATASETS_DIR, fps=30.0)
    supervisor.attach_recorder(recorder)
    app["bus"] = bus
    app["supervisor"] = supervisor
    app["recorder"] = recorder

    async def _on_startup(_: web.Application) -> None:
        bus.attach_loop(asyncio.get_running_loop())

    async def _on_cleanup(_: web.Application) -> None:
        supervisor.shutdown()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    routes = [
        web.get("/", serve_dashboard),
        web.get("/healthz", healthz),
        web.get("/v1/world", get_world),
        web.get("/v1/world/stream", stream_world),
        web.post("/v1/teleop/jog", post_jog),
        web.post("/v1/teleop/home", post_home),
        web.post("/v1/teleop/gripper", post_gripper),
        web.post("/v1/teleop/estop", post_estop),
        web.post("/v1/teleop/estop/clear", post_estop_clear),
        web.post("/v1/teleop/ghost", post_ghost_pose),
        web.get("/v1/cameras/{name}.jpg", get_camera_jpeg),
        web.post("/v1/cameras/swap", post_cameras_swap),
        web.post("/v1/teleop/drive_mode", post_drive_mode),
        web.get("/v1/teleop/filter", get_filter_params),
        web.post("/v1/teleop/filter", post_filter_params),
        web.post("/v1/recording/start", post_recording_start),
        web.post("/v1/recording/save", post_recording_save),
        web.post("/v1/recording/cancel", post_recording_cancel),
        web.get("/v1/recording/state", get_recording_state),
        web.get("/v1/hud", get_hud),
        web.get("/v1/episodes", list_episodes),
        web.get("/v1/episodes/{id}/meta.json", get_episode_meta),
        web.get("/v1/episodes/{id}/frames.jsonl", get_episode_frames),
        web.get("/v1/episodes/{id}/thumbnail.jpg", get_episode_thumbnail),
        web.get("/v1/episodes/{id}/cameras/{cam}/{idx}.jpg", get_episode_camera_frame),
        web.delete("/v1/episodes/{id}", delete_episode),
    ]
    for route in routes:
        app.router.add_route(route.method, route.path, route.handler)

    if WEB_DIR.is_dir():
        app.router.add_static("/web/", path=str(WEB_DIR), show_index=False)
    # Serve the URDF + STL meshes so the in-browser Three.js scene can load
    # them. The URDF references its meshes by relative path so this single
    # static route covers everything (uf850.urdf, meshes/visual/*.stl, etc.).
    if ASSETS_DIR.is_dir():
        app.router.add_static("/assets/", path=str(ASSETS_DIR), show_index=False)

    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=False, expose_headers="*",
                allow_headers="*", allow_methods="*",
            )
        },
    )
    for r in list(app.router.routes()):
        try:
            cors.add(r)
        except ValueError:
            # aiohttp_cors raises on the static route; skip it.
            pass

    return app


def run(
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    ros_port: int = 10000,
    backend: RobotBackend | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    app = build_app(backend=backend)

    from farm_edge_agent.ros_bridge import RosTcpBridge
    bridge = RosTcpBridge(supervisor=app["supervisor"], host=host, port=ros_port)
    bridge.start()
    app["ros_bridge"] = bridge
    app["supervisor"].attach_bridge(bridge)

    async def _stop_bridge(_: web.Application) -> None:
        bridge.stop()

    app.on_cleanup.append(_stop_bridge)

    backend_name = getattr(app["supervisor"].backend, "backend_name", "?")
    log.info(
        "farm edge daemon (%s): http://%s:%d  ·  ros-tcp tcp://%s:%d",
        backend_name, host, port, host, ros_port,
    )
    web.run_app(app, host=host, port=port, print=None)


__all__ = ["build_app", "run"]
