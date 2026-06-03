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
POST /v1/teleop/joints          — {joints: [..6 rad], gripper?: 0..1}
GET  /v1/policy/prompt          — current language prompt for the eval client
POST /v1/policy/prompt          — {prompt: str} (dashboard input field)
GET  /v1/policy/heartbeat       — last heartbeat the eval client posted
POST /v1/policy/heartbeat       — eval client → daemon: alive + policy server health
POST /v1/policy/run             — spawn model/eval_pi05.py (live by default)
POST /v1/policy/stop            — SIGTERM the eval subprocess
GET  /v1/policy/run/state       — {running, pid, exit_code?, args, log[]}
POST /v1/teleop/estop           — software emergency stop
POST /v1/teleop/estop/clear     — re-arm after e-stop
GET  /v1/teleop/filter          — current 1€ joint-filter params
POST /v1/teleop/filter          — {min_cutoff?, beta?} hot-tune 1€ filter
GET  /v1/cameras/{name}.jpg     — live JPEG from a backend camera (placeholder for xarm)
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
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

# parents: server → farm_edge_agent → src → edge-agent → teleop → repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
# URDF + STL meshes ship inside the package (edge-agent/assets/…).
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
# The dashboard lives at the repo-level ui/ folder; the daemon serves it from there.
UI_DIR = REPO_ROOT / "ui"
# Teleop recordings land here (gitignored, under the consolidated datasets/ dir).
DATASETS_DIR = REPO_ROOT / "datasets" / "dataset4"
EVAL_SCRIPT = REPO_ROOT / "model" / "eval_pi05.py"
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


def _is_local_origin(request: web.Request) -> bool:
    """CSRF guard for state-changing endpoints. A browser attaches an Origin
    header on cross-site requests; allow only when it's absent (curl / the eval
    client / same-origin GETs) or matches the daemon's own host. This blocks a
    foreign web page the operator has open from POSTing a run that moves the arm.
    """
    origin = request.headers.get("Origin")
    if not origin:
        return True
    from urllib.parse import urlparse

    host = (urlparse(origin).hostname or "").lower()
    req_host = (request.host or "").split(":")[0].lower()
    return host in ("127.0.0.1", "localhost", "::1") or host == req_host


# ── dev live-reload (opt-in via FARM_DEV_RELOAD=1) ─────────────────────────────
# When enabled, a middleware injects a tiny EventSource snippet into every served
# HTML page; the /v1/dev/livereload stream fires "reload" the moment any
# ui/*.html file changes on disk, so edits show instantly with no manual refresh.
# Off by default so the operator dashboard never auto-reloads during real use.

_RELOAD_SNIPPET = (
    b"<script>(function(){try{var e=new EventSource('/v1/dev/livereload');"
    b"e.onmessage=function(m){if(m.data==='reload')location.reload();};}catch(_){}"
    b"})();</script>"
)


def _ui_signature() -> str:
    """Cheap fingerprint of the served HTML files (name + mtime + size)."""
    parts: list[str] = []
    if UI_DIR.is_dir():
        for p in sorted(UI_DIR.glob("*.html")):
            try:
                st = p.stat()
                parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                pass
    return "|".join(parts)


@web.middleware
async def _dev_livereload_mw(request: web.Request, handler: Any) -> web.StreamResponse:
    resp = await handler(request)
    try:
        if (request.app.get("dev_reload")
                and isinstance(resp, web.Response)
                and (resp.content_type or "").startswith("text/html")
                and isinstance(resp.body, (bytes, bytearray))
                and b"</body>" in resp.body):
            resp.body = bytes(resp.body).replace(b"</body>", _RELOAD_SNIPPET + b"</body>", 1)
    except Exception:  # noqa: BLE001 — dev tooling must never break a page
        pass
    return resp


async def dev_livereload(request: web.Request) -> web.StreamResponse:
    """SSE that emits 'reload' whenever a ui/*.html file changes on disk."""
    if not request.app.get("dev_reload"):
        return web.Response(status=404)
    resp = web.StreamResponse(headers=SSE_HEADERS)
    await resp.prepare(request)
    sig = _ui_signature()
    await resp.write(b"data: connected\n\n")
    ticks = 0
    try:
        while not request.transport.is_closing():
            await asyncio.sleep(0.4)
            new = _ui_signature()
            if new != sig:
                sig = new
                await resp.write(b"data: reload\n\n")
            else:
                ticks += 1
                if ticks % 35 == 0:  # ~14s keepalive
                    await resp.write(b": ping\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


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


async def post_joint_target(request: web.Request) -> web.Response:
    """Joint-space sibling of ``/v1/teleop/ghost`` for learned-policy
    eval. Body: ``{"joints": [j1..j6 in radians], "gripper": optional 0-1}``.
    Bypasses IK — the joints are pushed straight into the same stream
    history buffer the Quest teleop path writes to (after IK), so the
    real arm follows when ``drive_real_arm`` is on."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "body must be JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    joints = body.get("joints")
    if not (isinstance(joints, list) and len(joints) == 6):
        return web.json_response(
            {"error": "body must include 'joints' as a list of 6 floats (radians)"},
            status=400,
        )
    try:
        joints = [float(j) for j in joints]
    except (TypeError, ValueError):
        return web.json_response({"error": "joints must be numeric"}, status=400)
    grip_raw = body.get("gripper")
    gripper: float | None = None
    if grip_raw is not None:
        try:
            gripper = float(grip_raw)
        except (TypeError, ValueError):
            return web.json_response(
                {"error": "gripper must be a number in [0, 1] or omitted"},
                status=400,
            )
    supervisor: Supervisor = request.app["supervisor"]
    result = await asyncio.to_thread(
        supervisor.set_joint_target, joints, gripper=gripper
    )
    if isinstance(result, dict) and "error" in result:
        return web.json_response(result, status=409)
    return web.json_response(result)


async def get_policy_prompt(request: web.Request) -> web.Response:
    """Return the latest language prompt set from the dashboard input.

    The prompt lives in app state (not on the backend) — it's a hint for
    the external eval client (``model/eval_pi05.py``), not arm state.
    Empty string means the dashboard hasn't set one yet; the eval client
    falls back to its ``--task`` CLI default in that case.
    """
    return web.json_response({"prompt": str(request.app.get("policy_prompt", ""))})


async def post_policy_prompt(request: web.Request) -> web.Response:
    """Set the language prompt the dashboard wants the eval client to use.

    Body: ``{"prompt": "<string>"}``. Empty string clears the override.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "body must be JSON"}, status=400)
    if not isinstance(body, dict) or "prompt" not in body:
        return web.json_response({"error": "body must include 'prompt'"}, status=400)
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        return web.json_response({"error": "prompt must be a string"}, status=400)
    # Trim incidental whitespace so the dashboard input doesn't accidentally
    # send a leading/trailing newline.
    prompt = prompt.strip()
    request.app["policy_prompt"] = prompt
    log.info("policy prompt set: %r", prompt)
    return web.json_response({"prompt": prompt})


async def get_policy_heartbeat(request: web.Request) -> web.Response:
    """Latest heartbeat the eval client posted, augmented with how stale
    it is. Dashboard polls this to colour the "eval running?" indicator
    without needing a websocket. Empty body if nothing posted yet.
    """
    hb = request.app.get("policy_heartbeat")
    if not hb:
        return web.json_response({"present": False})
    age = max(0.0, time.time() - float(hb.get("server_ts", 0.0)))
    return web.json_response({"present": True, "age_s": round(age, 2), **hb})


async def post_policy_heartbeat(request: web.Request) -> web.Response:
    """Eval client → daemon liveness ping. Optional fields document
    what the eval client is doing right now; the dashboard surfaces them.

    Body (all optional)::

        {
            "policy_url": str,
            "policy_ok": bool,
            "last_chunk_ms": float,
            "last_action_idx": int,
            "task_prompt": str,
            "drive_real_arm": bool,
            "dry_run": bool,
            "note": str,
        }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    body["server_ts"] = time.time()
    request.app["policy_heartbeat"] = body
    return web.json_response({"ok": True, "server_ts": body["server_ts"]})


def _drain_eval_output(proc: subprocess.Popen, log_deque: collections.deque[str]) -> None:
    """Background thread: copy the eval subprocess's stdout into a
    bounded deque so the dashboard can tail it. Closes when the process
    exits, which makes ``proc.stdout.readline()`` return ``""``."""
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            log_deque.append(line.rstrip("\n"))
    except Exception as exc:
        log_deque.append(f"<drain error: {exc}>")


async def post_policy_run(request: web.Request) -> web.Response:
    """Spawn ``model/eval_pi05.py`` as a subprocess. Body fields override
    defaults; all optional::

        {
            "policy_url":    "ws://127.0.0.1:8000",
            "live":          true,            # else --dry-run
            "mode":          "queue",         # default | queue | sync (execution loop)
            "rtc":           true,            # Real-Time Chunking seam smoothing
            "max_steps":     600,
            "steps_per_chunk": 3,
            "rate_hz":       5.0,
            "motion_scale":  0.25,
            "action_mode":   "absolute",
            "no_gripper":    false
        }

    Idempotent on the running state: returns 409 if already running."""
    proc = request.app.get("eval_process")
    if proc is not None and proc.poll() is None:
        return web.json_response(
            {"error": "already running", "pid": proc.pid}, status=409,
        )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    # If a cluster serve is being driven, make sure the two-hop tunnel is up
    # BEFORE the eval client tries to connect. The client now retries for ~45 s,
    # but bringing the relay + port-forward up first means Run just works instead
    # of burning that window (and avoids the "Run does nothing" confusion when the
    # tunnel had quietly torn down). Best-effort — never blocks the run.
    try:
        from farm_edge_agent.server import cluster
        serve_job = request.app.get("serve_job")
        if serve_job and cluster.available():
            state, node = await asyncio.to_thread(cluster.serve_state, serve_job["job_id"])
            if state == "RUNNING" and node:
                serve_job["node"] = node
                if not serve_job.get("socat"):
                    serve_job["socat"] = await asyncio.to_thread(cluster.serve_socat_up, node)
                await asyncio.to_thread(_serve_pf_start, request.app)
    except Exception as exc:
        log.warning("serve tunnel pre-check before run failed: %s", exc)

    args: list[str] = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--policy-url", str(body.get("policy_url", "ws://127.0.0.1:8000")),
        # 30 Hz native cadence + all 10 actions per chunk. This was
        # what the model was trained against; going faster trades
        # accuracy for speed (PD lag amplification).
        "--max-steps", str(int(body.get("max_steps", 1800))),
        "--steps-per-chunk", str(int(body.get("steps_per_chunk", 10))),
        # Native 30 Hz cadence + all 10 actions per chunk — the config the model
        # was trained against and that performed the task reliably. The
        # smoothness experiments (15 Hz playback, 100 Hz client interpolation,
        # RTC) regressed task performance when stacked, so they are OFF by
        # default and opt-in per request: pass {"stream_hz": 100, "rate_hz": 15}
        # and/or {"rtc": true} to re-enable, one at a time, with arm testing.
        "--rate-hz", str(float(body.get("rate_hz", 30.0))),
        "--stream-hz", str(float(body.get("stream_hz", 0.0))),
        "--motion-scale", str(float(body.get("motion_scale", 0.25))),
        "--action-mode", str(body.get("action_mode", "absolute")),
        # Don't re-POST motion_scale — the dashboard already owns it.
        "--no-daemon-motion-scale",
    ]
    args.append("--live" if body.get("live", True) else "--dry-run")
    if not body.get("rtc", False):
        args.append("--no-rtc")
    # Execution-loop selector (eval_pi05.py's newer executors). "default" = the
    # proven timed-waypoint run_loop; "queue" = pipelined in-order FIFO (smooth,
    # self-limiting, never skips); "sync" = strict blocking no-skip (judge the
    # model, not the harness). RTC seam-smoothing composes with any of them.
    mode = str(body.get("mode", "default")).lower()
    if mode == "queue":
        args.append("--queue")
    elif mode == "sync":
        args.append("--sync")
    if body.get("no_gripper"):
        args.append("--no-gripper")

    request.app["eval_log"].clear()
    request.app["eval_log"].append(f"$ {' '.join(args)}")
    try:
        proc = subprocess.Popen(  # noqa: S603
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,  # so SIGINT to daemon doesn't kill us mid-cleanup
        )
    except Exception as exc:
        return web.json_response({"error": f"spawn failed: {exc}"}, status=500)
    request.app["eval_process"] = proc
    request.app["eval_cmd"] = args
    threading.Thread(
        target=_drain_eval_output,
        args=(proc, request.app["eval_log"]),
        daemon=True, name="eval-drain",
    ).start()
    log.info("eval subprocess started · pid=%s · args=%s", proc.pid, args)
    return web.json_response({"ok": True, "pid": proc.pid, "args": args})


async def post_policy_stop(request: web.Request) -> web.Response:
    """Terminate the eval subprocess (if any). SIGTERM first; SIGKILL
    after a short grace period. The eval client's signal handler halts the
    policy loop cleanly WITHOUT tripping an e-stop — the arm is
    position-controlled, so it holds its last commanded pose. (Use the
    dashboard E-STOP button for a real emergency halt.)"""
    proc = request.app.get("eval_process")
    if proc is None or proc.poll() is not None:
        return web.json_response({"ok": True, "running": False, "note": "not running"})
    pid = proc.pid
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
    request.app["eval_log"].append(f"[daemon] terminated pid={pid}")
    log.info("eval subprocess stopped · pid=%s", pid)
    return web.json_response({"ok": True, "running": False, "pid": pid})


async def get_policy_run_state(request: web.Request) -> web.Response:
    """Lightweight poll the dashboard uses to colour the Run button +
    tail the last few log lines."""
    proc = request.app.get("eval_process")
    log_deque: collections.deque[str] = request.app.get("eval_log") or collections.deque()
    log_lines = list(log_deque)
    if proc is None:
        return web.json_response({"running": False, "pid": None, "log": log_lines})
    rc = proc.poll()
    if rc is None:
        return web.json_response({
            "running": True, "pid": proc.pid,
            "args": request.app.get("eval_cmd", []),
            "log": log_lines,
        })
    return web.json_response({
        "running": False, "pid": proc.pid, "exit_code": rc,
        "args": request.app.get("eval_cmd", []),
        "log": log_lines,
    })


# ── Serving the trained policy on the cluster (dashboard-driven) ───────────
# The serve runs on a cluster GPU and binds :8000 on its worker. To reach it
# from the laptop we relay through two hops: a login-pod ``socat`` (worker →
# login pod, stood up by server/cluster.py) and a laptop-side
# ``kubectl port-forward`` (login pod → localhost, the daemon-managed subprocess
# below). Once both are up, ws://127.0.0.1:8000 reaches the policy — the exact
# URL the eval client (model/eval_pi05.py) already defaults to.

def _serve_pf_running(app: web.Application) -> bool:
    proc = app.get("serve_pf_proc")
    return proc is not None and proc.poll() is None


def _serve_pf_start(app: web.Application) -> bool:
    """Start the laptop-side ``kubectl port-forward pod/<login-pod> 8000:8000``.
    Idempotent — no-op if already running. This is the CLAUDE.md-sanctioned
    'forward to my own pod' path; the login-pod socat handles pod → worker."""
    if _serve_pf_running(app):
        return True
    from farm_edge_agent.server import cluster
    pod = cluster._pod()
    if not pod:
        return False
    port = cluster.SERVE_PORT
    try:
        proc = subprocess.Popen(  # noqa: S603
            ["kubectl", "port-forward", "-n", cluster.NS, f"pod/{pod}", f"{port}:{port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        log.warning("serve port-forward spawn failed: %s", exc)
        return False
    app["serve_pf_proc"] = proc
    log.info("serve port-forward started · pid=%s · localhost:%s → pod/%s", proc.pid, port, pod)
    return True


def _serve_pf_stop(app: web.Application) -> None:
    proc = app.get("serve_pf_proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    app["serve_pf_proc"] = None


def _serve_reachable(port: int) -> bool:
    """Does a TCP connect to the forwarded local port succeed? (End-to-end the
    tunnel is live and the serve has bound its socket.)"""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.6):
            return True
    except Exception:
        return False


def _serve_phase(state: str, launched: bool, bound: bool) -> str:
    if state in ("PENDING", "CONFIGURING"):
        return "queued"
    if state in ("FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
        return "stopped"
    if state == "RUNNING":
        if bound:
            return "serving"      # websocket server actually accepting → ready for Run
        if launched:
            return "loading"      # serve_policy launched, restoring params / JIT
        return "starting"         # container build / boot on the worker
    return "starting"


async def post_serve_start(request: web.Request) -> web.Response:
    """Submit the policy-serve sbatch on the cluster. Body (all optional)::

        {"model": "lora_gse", "step": "9999"}

    The tunnel is stood up lazily by /v1/serve/status once the job is RUNNING."""
    from farm_edge_agent.server import cluster
    if not cluster.available():
        return web.json_response({"error": "kubectl not available on this host"}, status=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    model = str(body.get("model", "lora_gse"))
    step = body.get("step")
    cur = request.app.get("serve_job")
    if cur:
        state, _ = await asyncio.to_thread(cluster.serve_state, cur["job_id"])
        if state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
            return web.json_response({"error": "serve already active", "job": cur}, status=409)
    res = await asyncio.to_thread(cluster.serve_launch, model, step)
    if "error" in res:
        return web.json_response(res, status=500)
    res.update(started_at=time.time(), node="", socat=False)
    request.app["serve_job"] = res
    log.info("serve job submitted · %s", res)
    return web.json_response({"ok": True, **res})


async def get_serve_status(request: web.Request) -> web.Response:
    """Lifecycle poll the dashboard uses to drive the serve control: SLURM
    state + tunnel health, and it lazily stands up the relay once RUNNING."""
    from farm_edge_agent.server import cluster
    app = request.app
    if not cluster.available():
        return web.json_response({"running": False, "kubectl": False})
    job = app.get("serve_job")
    if job is None:
        found = await asyncio.to_thread(cluster.serve_discover)
        if found is None:
            return web.json_response({"running": False, "kubectl": True})
        found.update(started_at=time.time(), node="", socat=False)
        app["serve_job"] = job = found
    job_id = job["job_id"]
    state, node = await asyncio.to_thread(cluster.serve_state, job_id)
    if state == "GONE":
        _serve_pf_stop(app)
        await asyncio.to_thread(cluster.serve_socat_down)
        app["serve_job"] = None
        return web.json_response({"running": False, "kubectl": True, "state": "GONE", "job_id": job_id})
    if node and not job.get("node"):
        job["node"] = node
    tail = await asyncio.to_thread(cluster.serve_log_tail, job_id, 30)
    launched, bound = await asyncio.to_thread(cluster.serve_markers, job_id)
    # Latch bound: a serve stays accepting once bound (until the job ends), so
    # never let a transient log/grep miss flip it back to "starting".
    bound = bool(job.get("bound") or bound)
    job["bound"] = bound
    # Once the worker is known, stand up the relay + laptop forward (idempotent),
    # so the tunnel is already in place by the time the serve binds.
    if state == "RUNNING" and node:
        if not job.get("socat"):
            job["socat"] = await asyncio.to_thread(cluster.serve_socat_up, node)
        await asyncio.to_thread(_serve_pf_start, app)
    # NB: a bare TCP probe to the forwarded port lies (kubectl's local listener
    # accepts before the upstream serve exists), so phase is driven by the log's
    # bound marker, not reachability. reachable is reported only as a hint.
    reachable = (await asyncio.to_thread(_serve_reachable, cluster.SERVE_PORT)
                 if _serve_pf_running(app) else False)
    return web.json_response({
        "running": True, "kubectl": True, "job_id": job_id,
        "model": job.get("model"), "step": job.get("step"),
        "state": state, "node": node, "phase": _serve_phase(state, launched, bound),
        "bound": bound, "launched": launched,
        "socat": bool(job.get("socat")), "port_forward": _serve_pf_running(app),
        "reachable": reachable,
        "elapsed_s": round(time.time() - job.get("started_at", time.time())),
        "log_tail": "\n".join(tail.strip().splitlines()[-12:]),
    })


async def post_serve_stop(request: web.Request) -> web.Response:
    """Cancel the serve job and tear down both tunnel hops."""
    from farm_edge_agent.server import cluster
    app = request.app
    job = app.get("serve_job")
    _serve_pf_stop(app)
    if job is None:
        await asyncio.to_thread(cluster.serve_socat_down)
        return web.json_response({"ok": True, "running": False, "note": "no tracked job"})
    res = await asyncio.to_thread(cluster.serve_stop, job["job_id"])
    app["serve_job"] = None
    log.info("serve job stopped · %s", job.get("job_id"))
    return web.json_response({"ok": res.get("ok", True), "running": False, "job_id": job["job_id"]})


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


async def get_motion_scale(request: web.Request) -> web.Response:
    """Current controller→arm motion-scale ratio (1.0 = 1:1)."""
    bridge = request.app.get("ros_bridge")
    if bridge is None or not hasattr(bridge, "motion_scale"):
        return web.json_response({"error": "ros bridge unavailable"}, status=503)
    return web.json_response({"scale": float(bridge.motion_scale)})


async def post_motion_scale(request: web.Request) -> web.Response:
    """Set the controller→arm motion-scale ratio. Body: ``{"scale": <float>}``."""
    bridge = request.app.get("ros_bridge")
    if bridge is None or not hasattr(bridge, "motion_scale"):
        return web.json_response({"error": "ros bridge unavailable"}, status=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict) or "scale" not in body:
        return web.json_response({"error": "body must include 'scale'"}, status=400)
    try:
        scale = float(body["scale"])
    except (TypeError, ValueError):
        return web.json_response({"error": "scale must be a number"}, status=400)
    try:
        bridge.motion_scale = scale
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"scale": float(bridge.motion_scale)})


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
        desc = meta.get("description")
        out.append({
            "id": p.name,
            "start_iso": meta.get("start_iso"),
            "duration_s": meta.get("duration_s"),
            "fps": meta.get("fps"),
            "frame_count": meta.get("frame_count"),
            "cameras": list(cameras),
            "backend": meta.get("backend"),
            "has_description": bool(desc and str(desc).strip()),
            "description": desc if isinstance(desc, str) else None,
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


def _camera_timing_sync(p: Path) -> dict[str, Any]:
    """Per-camera timing for the review app's colorbars.

    For each camera, walks ``frames.jsonl`` and, at each row, decides
    whether that frame's JPEG is a *fresh* capture by checking:

    * the JPEG file exists on disk (recorder couldn't write None blobs)
    * its size differs from the previous fresh frame's size (the cam
      subprocess cache returns identical bytes on a hiccup, so a size
      match is a strong duplicate signal)

    Returns ``{"fps_target": <hz>, "cameras": {<name>: {"dt_ms": [...]}}}``.
    """
    meta_path = p / "meta.json"
    frames_path = p / "frames.jsonl"
    if not meta_path.is_file() or not frames_path.is_file():
        return {"fps_target": 30.0, "cameras": {}}
    meta = json.loads(meta_path.read_text())
    fps = float(meta.get("fps") or 30.0)
    target_ms = 1000.0 / fps
    cam_root = p / "cameras"
    cameras = list(meta.get("cameras") or [])
    if not cameras and cam_root.is_dir():
        cameras = sorted(c.name for c in cam_root.iterdir() if c.is_dir())

    # Load frame timestamps (just t and frame index — skip the rest).
    rows: list[tuple[int, float]] = []
    with frames_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows.append((int(row.get("frame", -1)), float(row.get("t", 0.0))))
            except Exception:
                continue

    out_cams: dict[str, dict[str, list[float]]] = {}
    for cam in cameras:
        cdir = cam_root / cam
        dt_ms: list[float] = []
        last_fresh_t: float | None = None
        last_size: int | None = None
        for idx, t in rows:
            sz = -1
            try:
                sz = (cdir / f"{idx:06d}.jpg").stat().st_size
            except (FileNotFoundError, OSError):
                pass
            is_fresh = sz > 0 and sz != last_size
            if is_fresh:
                dt = target_ms if last_fresh_t is None else (t - last_fresh_t) * 1000.0
                dt_ms.append(round(dt, 2))
                last_fresh_t = t
                last_size = sz
            else:
                # Stale or missing tick: report the growing gap so the
                # colorbar shows red until a fresh frame arrives.
                gap_ms = (
                    (t - last_fresh_t) * 1000.0
                    if last_fresh_t is not None
                    else target_ms * 100
                )
                dt_ms.append(round(gap_ms, 2))
        out_cams[cam] = {"dt_ms": dt_ms}
    return {"fps_target": fps, "cameras": out_cams}


async def get_episode_camera_timing(request: web.Request) -> web.Response:
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.json_response({"error": "not found"}, status=404)
    data = await asyncio.to_thread(_camera_timing_sync, p)
    return web.json_response(data)


async def patch_episode_meta(request: web.Request) -> web.Response:
    """Update whitelisted fields in an episode's ``meta.json``.

    Currently only the free-text ``description`` is editable from the
    review UI. Other meta fields are derived from the recording and
    should not be hand-edited.
    """
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.json_response({"error": "not found"}, status=404)
    meta_path = p / "meta.json"
    if not meta_path.is_file():
        return web.json_response({"error": "no meta"}, status=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)

    def _write() -> dict[str, Any]:
        meta = json.loads(meta_path.read_text())
        if "description" in body:
            desc = body["description"]
            if desc is None:
                meta.pop("description", None)
            else:
                meta["description"] = str(desc)
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    meta = await asyncio.to_thread(_write)
    return web.json_response(meta)


async def clip_episode(request: web.Request) -> web.Response:
    """Destructively trim an episode to the half-open frame range
    ``[clip_in, clip_out)``. Drops out-of-range JSONL rows and JPEGs,
    renumbers what remains starting at 0, and rewrites ``meta.json``
    with the new frame count + duration. Refuses to operate on an
    in-progress recording."""
    p = _episode_dir(request.match_info["id"])
    if p is None:
        return web.json_response({"error": "not found"}, status=404)
    rec = request.app["supervisor"].recorder
    if rec is not None:
        st = rec.state
        if st.get("recording") and st.get("episode_id") == p.name:
            return web.json_response(
                {"error": "cannot clip an episode that is still recording"},
                status=409,
            )
    meta_path = p / "meta.json"
    frames_path = p / "frames.jsonl"
    if not meta_path.is_file() or not frames_path.is_file():
        return web.json_response({"error": "missing meta or frames"}, status=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        clip_in = int(body["clip_in"])
        clip_out = int(body["clip_out"])
    except (KeyError, TypeError, ValueError):
        return web.json_response(
            {"error": "body must include integer clip_in and clip_out"}, status=400
        )

    def _clip() -> dict[str, Any]:
        meta = json.loads(meta_path.read_text())
        total = int(meta.get("frame_count", 0))
        if not (0 <= clip_in < clip_out <= total):
            raise ValueError(
                f"clip range [{clip_in},{clip_out}) out of bounds for {total} frames"
            )
        if clip_in == 0 and clip_out == total:
            return meta  # nothing to do

        # Pass 1: filter frames.jsonl, re-index, rebase t.
        kept: list[dict[str, Any]] = []
        base_t: float | None = None
        with frames_path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = int(row.get("frame", -1))
                if not (clip_in <= idx < clip_out):
                    continue
                if base_t is None:
                    base_t = float(row.get("t", 0.0))
                row["frame"] = idx - clip_in
                row["t"] = round(float(row.get("t", 0.0)) - (base_t or 0.0), 4)
                kept.append(row)
        # Atomic-ish rewrite via .tmp swap so a crash mid-write doesn't
        # leave the episode unreadable.
        tmp = frames_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for row in kept:
                fh.write(json.dumps(row) + "\n")
        tmp.replace(frames_path)

        # Pass 2: prune + rename camera JPEGs. New_idx = old_idx - clip_in
        # is always <= old_idx, so renaming low → high never collides.
        cam_root = p / "cameras"
        cameras = meta.get("cameras") or []
        for cam in cameras:
            cdir = cam_root / cam
            if not cdir.is_dir():
                continue
            files = sorted(f for f in cdir.iterdir() if f.is_file())
            for f in files:
                try:
                    idx = int(f.stem)
                except ValueError:
                    continue
                if not (clip_in <= idx < clip_out):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                    continue
                new_name = f"{idx - clip_in:06d}.jpg"
                if f.name != new_name:
                    f.rename(cdir / new_name)

        # Update meta.
        meta["frame_count"] = len(kept)
        meta["duration_s"] = round(float(kept[-1]["t"]) if kept else 0.0, 3)
        clips = list(meta.get("clip_history") or [])
        clips.append({"clip_in": clip_in, "clip_out": clip_out, "kept": len(kept)})
        meta["clip_history"] = clips
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    try:
        meta = await asyncio.to_thread(_clip)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        log.warning("episode clip failed for %s: %s", p, exc)
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(meta)


async def serve_review(_: web.Request) -> web.Response:
    review = UI_DIR / "review.html"
    if not review.is_file():
        return web.Response(text="review app missing (ui/review.html)", status=500)
    return web.Response(body=review.read_bytes(), content_type="text/html")


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
    index = UI_DIR / "index.html"
    if not index.is_file():
        return web.Response(text="dashboard missing (ui/index.html)", status=500)
    return web.Response(body=index.read_bytes(), content_type="text/html")


# ── training (cluster bridge) ────────────────────────────────────────────────


async def serve_train(_: web.Request) -> web.Response:
    page = UI_DIR / "train.html"
    if not page.is_file():
        return web.Response(text="train page missing (ui/train.html)", status=500)
    return web.Response(body=page.read_bytes(), content_type="text/html")


async def get_train_models(_: web.Request) -> web.Response:
    """The three architectures + their default steps/gpus, for the config form."""
    from farm_edge_agent.server import cluster
    models = {
        k: {"label": v["label"], "config": v["config"], "steps": v["steps"], "gpus": v["gpus"]}
        for k, v in cluster.MODELS.items()
    }
    return web.json_response({"models": models, "kubectl": cluster.available()})


async def post_train_launch(request: web.Request) -> web.Response:
    from farm_edge_agent.server import cluster
    if request.app.get("train_job") is not None:
        phase = (request.app.get("train_status_cache") or {}).get("data", {}).get("phase")
        if phase in ("queued", "starting", "training"):
            return web.json_response(
                {"error": "a job is already running"}, status=409,
            )
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = str(body.get("model", "gse"))
    spec = cluster.MODELS.get(model)
    if spec is None:
        return web.json_response({"error": f"unknown model {model!r}"}, status=400)
    steps = int(body.get("steps", spec["steps"]))
    gpus = int(body.get("gpus", spec["gpus"]))
    result = await asyncio.to_thread(cluster.launch, model, steps, gpus)
    if "error" in result:
        return web.json_response(result, status=502)
    result["total_steps"] = steps
    result["started_at"] = time.time()
    request.app["train_job"] = result
    request.app["train_status_cache"] = {}
    log.info("training launched · %s · job %s · %d steps · %d gpu", model, result["job_id"], steps, gpus)
    return web.json_response(result)


async def get_train_status(request: web.Request) -> web.Response:
    from farm_edge_agent.server import cluster
    job = request.app.get("train_job")
    if job is None:
        # Adopt a running job so the page reflects the cluster after a daemon
        # restart (or an out-of-band sbatch). Opt-in via FARM_CLUSTER_ADOPT so
        # tests and ad-hoc `build_app()` callers never shell out to kubectl.
        if os.environ.get("FARM_CLUSTER_ADOPT") and cluster.available():
            job = await asyncio.to_thread(cluster.discover)
        if job is None:
            return web.json_response({"active": False, "kubectl": cluster.available()})
        request.app["train_job"] = job
        request.app["train_status_cache"] = {}
        log.info("adopted running job %s (%s, %d steps)", job["job_id"], job["model"], job["total_steps"])
    # Rate-limit the kubectl polling (shared cache) so many browser tabs / a
    # fast poll interval don't hammer the pod.
    cache = request.app.get("train_status_cache") or {}
    now = time.monotonic()
    if cache.get("at") and now - cache["at"] < 2.5:
        return web.json_response(cache["data"])
    data = await asyncio.to_thread(
        cluster.status, job["job_id"], job["model"], job["total_steps"]
    )
    data.update(active=True, started_at=job.get("started_at"),
                gpus=job.get("gpus"), config=job.get("config"))
    # Auto-release the GPUs so a finished/hung run can't keep an allocation.
    # Normal completion ends the SLURM job on its own (which frees the GPUs);
    # this only scancels if training has clearly started logging steps and then
    # made NO progress for 20 min — i.e. it finished or hung. Container build +
    # norm-stats (no steps yet) and the post-training checkpoint push are both
    # well inside the grace, so a healthy run is never cut short.
    if data.get("phase") == "training":
        step = data.get("step", 0)
        total = data.get("total_steps", 0) or 0
        if step > job.get("_last_step", -1):
            job["_last_step"] = step
            job["_progress_at"] = now
        # Only cut a genuinely-stalled run mid-training. A frozen step in the
        # final stretch is the post-training drain (final checkpoint save + HF
        # push), not a hang — scancelling there would kill the final upload.
        elif (job.get("_progress_at") and now - job["_progress_at"] > 1200
              and step < 0.95 * total):
            await asyncio.to_thread(cluster.stop, job["job_id"])
            data["phase"] = "stopped"
            data["note"] = "auto-stopped: no step progress for 20 min — GPUs released"
            log.info("auto-stopped stalled training job %s", job["job_id"])
    request.app["train_status_cache"] = {"at": now, "data": data}
    return web.json_response(data)


async def post_train_stop(request: web.Request) -> web.Response:
    from farm_edge_agent.server import cluster
    job = request.app.get("train_job")
    if job is None:
        return web.json_response({"ok": True, "note": "no active job"})
    result = await asyncio.to_thread(cluster.stop, job["job_id"])
    request.app["train_status_cache"] = {}
    return web.json_response(result)


async def get_train_metrics(request: web.Request) -> web.Response:
    """Per-GPU utilization + CPU load for the active job (only while running).

    Separate from /status (and more heavily rate-limited) because the
    ``srun --overlap`` into the job's node costs a second or two — keeping it
    off /status leaves the loss curve responsive.
    """
    from farm_edge_agent.server import cluster
    job = request.app.get("train_job")
    if job is None:
        return web.json_response({"active": False})
    # Only meaningful once the job is actually running.
    phase = (request.app.get("train_status_cache") or {}).get("data", {}).get("phase")
    if phase in ("queued", "done", "stopped"):
        return web.json_response({"active": True, "phase": phase, "gpus": [], "cpu": {}})
    cache = request.app.get("train_metrics_cache") or {}
    now = time.monotonic()
    if cache.get("at") and now - cache["at"] < 4.0:
        return web.json_response(cache["data"])
    data = await asyncio.to_thread(cluster.metrics, job["job_id"])
    data["active"] = True
    request.app["train_metrics_cache"] = {"at": now, "data": data}
    return web.json_response(data)


# ── consumer agent (/user page) ───────────────────────────────────────────────
# The /user page is the consumer-facing surface: type a high-level task, watch
# the agent look at the scene (DO vision), plan it (GPT-5.5), and execute it by
# hot-swapping per-object LoRA skills on the resident FFT-56k base. The
# orchestrator (server/agent.py) runs the pipeline and publishes events over SSE.


async def serve_user(_: web.Request) -> web.Response:
    page = UI_DIR / "user.html"
    if not page.is_file():
        return web.Response(text="user page missing (ui/user.html)", status=500)
    return web.Response(body=page.read_bytes(), content_type="text/html")


async def get_agent_config(_: web.Request) -> web.Response:
    from farm_edge_agent.server import agent
    return web.json_response(agent.agent_config())


async def get_agent_samples(_: web.Request) -> web.Response:
    """Recorded episodes the /user page can use as stand-in camera footage."""
    from farm_edge_agent.server import samples
    return web.json_response({"samples": await asyncio.to_thread(samples.list_samples)})


async def post_agent_run(request: web.Request) -> web.Response:
    """Start an agent run. Body (all optional except task)::

        {"task": str, "base_model": "fft_hotswap"|"fft",
         "skills": ["bottle","bear",...], "thinking_model": str,
         "vision_model": str, "execute": true, "step_seconds": 14}

    409 if a run is already in progress."""
    if not _is_local_origin(request):
        return web.json_response({"error": "cross-origin requests are not allowed"}, status=403)
    orch = request.app["orchestrator"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    if not str(body.get("task", "")).strip() and not str(body.get("ability_id", "")).strip():
        return web.json_response({"error": "body must include 'task' or 'ability_id'"}, status=400)
    try:
        orch.start(body)
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    return web.json_response({"ok": True})


async def post_agent_stop(request: web.Request) -> web.Response:
    orch = request.app["orchestrator"]
    await orch.stop()
    return web.json_response({"ok": True})


async def post_agent_execute(request: web.Request) -> web.Response:
    """Release the plan-then-wait gate so execution starts (the page's Run on arm).
    Optional body ``{"steps": [...], "confirm_threshold": 0..1}`` runs the
    user-edited plan and confirmation sensitivity."""
    if not _is_local_origin(request):
        return web.json_response({"error": "cross-origin requests are not allowed"}, status=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    orch = request.app["orchestrator"]
    ok = orch.continue_run(steps=body.get("steps"), confirm_threshold=body.get("confirm_threshold"))
    return web.json_response({"ok": ok})


async def get_agent_state(request: web.Request) -> web.Response:
    orch = request.app["orchestrator"]
    return web.json_response({"running": orch.running, "state": orch.state})


async def stream_agent(request: web.Request) -> web.StreamResponse:
    """SSE of orchestrator events. Replays the current run's history on connect
    so a page that loads mid-run catches up, then streams live."""
    orch = request.app["orchestrator"]
    bus = orch.bus
    resp = web.StreamResponse(headers=SSE_HEADERS)
    await resp.prepare(request)
    # Subscribe BEFORE snapshotting the replay so an event published during the
    # replay writes still lands in the queue; dedupe the overlap by monotonic seq.
    q = await bus.subscribe()
    last_seq = 0
    for event in bus.replay():
        await resp.write(_sse(event))
        last_seq = max(last_seq, event.get("seq", 0))
    try:
        while not request.transport.is_closing():
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
            except TimeoutError:
                await resp.write(b": ping\n\n")
                continue
            if event.get("seq", 0) <= last_seq:
                continue  # already delivered during replay
            last_seq = event["seq"]
            await resp.write(_sse(event))
    finally:
        bus.unsubscribe(q)
    return resp


# ── abilities (saved, reusable workflows) ─────────────────────────────────────


async def get_abilities(_: web.Request) -> web.Response:
    from farm_edge_agent.server import abilities
    return web.json_response({"abilities": abilities.list_abilities()})


async def post_ability(request: web.Request) -> web.Response:
    """Save a generated workflow as a reusable Ability. Body is the workflow
    ``{name, task, base_model, summary, steps:[...]}``."""
    if not _is_local_origin(request):
        return web.json_response({"error": "cross-origin requests are not allowed"}, status=403)
    from farm_edge_agent.server import abilities
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    try:
        rec = await asyncio.to_thread(abilities.save_ability, body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"ok": True, "ability": rec})


async def get_ability_detail(request: web.Request) -> web.Response:
    from farm_edge_agent.server import abilities
    ab = abilities.get_ability(request.match_info["id"])
    if ab is None:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response(ab)


async def delete_ability(request: web.Request) -> web.Response:
    if not _is_local_origin(request):
        return web.json_response({"error": "cross-origin requests are not allowed"}, status=403)
    from farm_edge_agent.server import abilities
    ok = await asyncio.to_thread(abilities.delete_ability, request.match_info["id"])
    return web.json_response({"ok": ok}, status=200 if ok else 404)


# ── wiring ──────────────────────────────────────────────────────────────────


def build_app(*, backend: RobotBackend | None = None) -> web.Application:
    """Build the aiohttp app. If ``backend`` is omitted, the sim backend
    is used (so existing tests keep working). Production callers pass
    a configured backend (SimBackend or XArmBackend) explicitly."""
    if backend is None:
        from farm_edge_agent.backends import SimBackend
        backend = SimBackend()

    app = web.Application(client_max_size=2_000_000, middlewares=[_dev_livereload_mw])
    # Dev live-reload: inject the reload watcher + enable /v1/dev/livereload when
    # FARM_DEV_RELOAD is set. Never on for the operator dashboard by default.
    app["dev_reload"] = bool(os.environ.get("FARM_DEV_RELOAD"))
    bus = EventBus(history=400)
    supervisor = Supervisor(bus, backend=backend)
    from farm_edge_agent.recorder import Recorder

    recorder = Recorder(supervisor, datasets_dir=DATASETS_DIR, fps=30.0)
    supervisor.attach_recorder(recorder)
    app["bus"] = bus
    app["supervisor"] = supervisor
    app["recorder"] = recorder
    # Live-typed language prompt for the eval client (model/eval_pi05.py).
    # See get_policy_prompt / post_policy_prompt.
    app["policy_prompt"] = ""
    # Most recent eval-client heartbeat (raw dict + server_ts). See the
    # heartbeat handlers; dashboard polls this for the eval-panel indicators.
    app["policy_heartbeat"] = {}
    # Daemon-managed eval subprocess (see post_policy_run/stop). The
    # process owns its own lifecycle; we just keep a handle so we can
    # report state and stop it on demand.
    app["eval_process"] = None
    app["eval_cmd"] = []
    app["eval_log"] = collections.deque(maxlen=300)
    # Active cluster training job (see /train page + server/cluster.py) and a
    # short-lived cache of its last polled status (rate-limits kubectl).
    app["train_job"] = None
    app["train_status_cache"] = {}
    app["train_metrics_cache"] = {}
    # Cluster policy-serve job (see /v1/serve/* + server/cluster.py) and the
    # laptop-side `kubectl port-forward` subprocess that relays it to localhost.
    app["serve_job"] = None
    app["serve_pf_proc"] = None
    # Consumer-agent orchestrator (/user page): scene → plan → skill-swap exec.
    from farm_edge_agent.server.agent import Orchestrator
    app["orchestrator"] = Orchestrator(app)

    async def _on_startup(_: web.Application) -> None:
        bus.attach_loop(asyncio.get_running_loop())

    async def _on_cleanup(_: web.Application) -> None:
        # Stop any in-flight agent run (which also stops its policy subprocess).
        orch = app.get("orchestrator")
        if orch is not None and orch.running:
            with contextlib.suppress(Exception):
                await orch.stop()
        # Bring down the eval subprocess if it's still running so a
        # daemon exit doesn't leave the policy commanding the arm.
        proc = app.get("eval_process")
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Drop the laptop-side serve port-forward (local). The GPU serve job +
        # login-pod socat are left running so a daemon restart re-adopts them
        # (stop the job explicitly from the dashboard to free the GPU).
        _serve_pf_stop(app)
        supervisor.shutdown()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    routes = [
        web.get("/", serve_dashboard),
        web.get("/review", serve_review),
        web.get("/train", serve_train),
        web.get("/user", serve_user),
        web.get("/v1/agent/config", get_agent_config),
        web.get("/v1/agent/samples", get_agent_samples),
        web.post("/v1/agent/run", post_agent_run),
        web.post("/v1/agent/stop", post_agent_stop),
        web.post("/v1/agent/execute", post_agent_execute),
        web.get("/v1/agent/state", get_agent_state),
        web.get("/v1/agent/stream", stream_agent),
        web.get("/v1/abilities", get_abilities),
        web.post("/v1/abilities", post_ability),
        web.get("/v1/abilities/{id}", get_ability_detail),
        web.delete("/v1/abilities/{id}", delete_ability),
        web.get("/v1/dev/livereload", dev_livereload),
        web.get("/v1/train/models", get_train_models),
        web.post("/v1/train/launch", post_train_launch),
        web.get("/v1/train/status", get_train_status),
        web.get("/v1/train/metrics", get_train_metrics),
        web.post("/v1/train/stop", post_train_stop),
        web.get("/healthz", healthz),
        web.get("/v1/world", get_world),
        web.get("/v1/world/stream", stream_world),
        web.post("/v1/teleop/jog", post_jog),
        web.post("/v1/teleop/home", post_home),
        web.post("/v1/teleop/gripper", post_gripper),
        web.post("/v1/teleop/estop", post_estop),
        web.post("/v1/teleop/estop/clear", post_estop_clear),
        web.post("/v1/teleop/ghost", post_ghost_pose),
        web.post("/v1/teleop/joints", post_joint_target),
        web.get("/v1/policy/prompt", get_policy_prompt),
        web.post("/v1/policy/prompt", post_policy_prompt),
        web.get("/v1/policy/heartbeat", get_policy_heartbeat),
        web.post("/v1/policy/heartbeat", post_policy_heartbeat),
        web.post("/v1/policy/run", post_policy_run),
        web.post("/v1/policy/stop", post_policy_stop),
        web.get("/v1/policy/run/state", get_policy_run_state),
        web.post("/v1/serve/start", post_serve_start),
        web.get("/v1/serve/status", get_serve_status),
        web.post("/v1/serve/stop", post_serve_stop),
        web.get("/v1/cameras/{name}.jpg", get_camera_jpeg),
        web.post("/v1/cameras/swap", post_cameras_swap),
        web.post("/v1/teleop/drive_mode", post_drive_mode),
        web.get("/v1/teleop/filter", get_filter_params),
        web.post("/v1/teleop/filter", post_filter_params),
        web.get("/v1/teleop/motion_scale", get_motion_scale),
        web.post("/v1/teleop/motion_scale", post_motion_scale),
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
        web.patch("/v1/episodes/{id}/meta", patch_episode_meta),
        web.post("/v1/episodes/{id}/clip", clip_episode),
        web.get("/v1/episodes/{id}/camera_timing", get_episode_camera_timing),
    ]
    for route in routes:
        app.router.add_route(route.method, route.path, route.handler)

    if UI_DIR.is_dir():
        app.router.add_static("/ui/", path=str(UI_DIR), show_index=False)
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
