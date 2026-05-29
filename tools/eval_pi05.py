"""Eval client for the trained pi 0.5 policy on the FARM UF850.

Connects to two services:

* the FARM edge daemon (``farm serve``) over HTTP for observations + actuation
* an openpi ``serve_policy.py`` instance over WebSocket for inference

Loop::

    obs ── GET /v1/world  +  GET /v1/cameras/{base,wrist}.jpg ──┐
                                                                 ▼
            policy.infer({state, image, prompt}) ── action chunk (H, 32)
                                                                 │
                                                                 ▼
    for i in range(steps_per_chunk):
        target_joints[i] = current_joints + Σ scale·delta[:6]   (joint-delta mode)
        POST /v1/teleop/joints {joints, gripper}                (or print in --dry-run)
        sleep(1/rate_hz)

Defaults are conservative: ``--dry-run`` is implicit unless ``--live`` is
passed, ``--motion-scale 0.25`` (client-side delta scale), and the
script refuses to send live commands until ``drive_real_arm`` has been
flipped on from the dashboard.

────────────────────────────────────────────────────────────────────────
Cluster-side serve_policy.py setup
────────────────────────────────────────────────────────────────────────

The policy runs on the CS153 cluster (login pod is `slurm-login-nhweiss-*`)
and the eval client (this script) runs on the laptop. Connect them with
`kubectl port-forward`.

1. (Once) install openpi-client into the login pod's python env so we
   know the WebSocket protocol matches::

       kubectl exec -it -n slurm <pod> -c login -- runuser -u nhweiss -- bash -lc \\
           'cd /home/nhweiss/farm-train/openpi && pip install -e packages/openpi-client'

2. Pull a trained checkpoint into a stable path the srun job can read.
   The full fine-tune keeps step-5000 … step-19999; step-19999 is the
   final one (openpi names the final checkpoint at step N-1). Swap in an
   earlier tag to compare::

       kubectl exec -it -n slurm <pod> -c login -- runuser -u nhweiss -- bash -lc \\
           'hf download NoahWeiss/farm_uf850_pi05 \\
              --include "step-19999/*" --local-dir ~/farm-train/checkpoints/pi05_step19999'

3. Submit serve_policy.py to a small partition. Single H100 is plenty
   for batch-1 inference. NOTE: the openpi WebSocket port is 8000 by
   default; this script's ``--policy-url`` default matches that::

       cat > ~/farm-train/serve_pi05.sbatch <<'EOF'
       #!/bin/bash
       #SBATCH --partition=small
       #SBATCH --gres=gpu:1
       #SBATCH --cpus-per-task=16
       #SBATCH --time=04:00:00
       #SBATCH --output=serve-pi05-%j.out
       #SBATCH --job-name=serve-pi05

       cd /home/nhweiss/farm-train/openpi
       srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \\
            --container-mounts=/home/nhweiss:/home/nhweiss \\
            uv run python scripts/serve_policy.py \\
                --env=ALOHA_SIM \\
                policy:checkpoint \\
                --policy.config=pi05_farm_uf850 \\
                --policy.dir=/home/nhweiss/farm-train/checkpoints/pi05_step19999/step-19999 \\
                --port=8000
       EOF
       sbatch ~/farm-train/serve_pi05.sbatch

4. Discover which worker the job landed on and forward its port to the
   login pod, then from the laptop forward the login pod's port locally::

       # In the login pod:
       JOB=$(squeue -u $USER -h -o '%i' | head -1)
       WORKER=$(squeue -j $JOB -h -o '%R')
       kubectl port-forward -n slurm pod/$WORKER 8000:8000 &

       # On the laptop:
       kubectl port-forward -n slurm <my-login-pod> 8000:8000

5. Run this script::

       python tools/eval_pi05.py --task "Picking up the bottle and placing it on the box" \\
           --policy-url ws://127.0.0.1:8000 \\
           --dry-run

Swap to step-10000 (once training finishes) by re-running step 2 with
the newer tag and pointing ``--policy.dir`` at the new directory.
"""

from __future__ import annotations

import argparse
import io
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_DAEMON_URL = "http://127.0.0.1:8787"
DEFAULT_POLICY_URL = "ws://127.0.0.1:8000"
DEFAULT_TASK = "Picking up the bottle and placing it on the box"
# Commit at the dataset's native 30 Hz (each policy action ≈ state at t+1/30s).
# This is the cadence the model was trained against and that performs the task
# reliably. The streaming/RTC smoothness path (--stream-hz, --rtc) is off by
# default — it improved smoothness but regressed task performance when stacked,
# so re-add it deliberately and test on the arm.
DEFAULT_RATE_HZ = 30.0
# Action chunk = 10 actions per inference (π0.5 is hardcoded at action_horizon=10).
# Consume all 10 each chunk to maximise motion duty cycle vs. inference overhead.
DEFAULT_STEPS_PER_CHUNK = 10
DEFAULT_MOTION_SCALE = 0.25
# 60 s of action time. Click Stop on the dashboard early if a re-task
# is needed.
DEFAULT_MAX_STEPS = 1800
DEFAULT_IMAGE_SHAPE = (224, 224)    # pi 0.5 paligemma vision tower input

TASKS_PATH = Path(__file__).resolve().parent.parent / "datasets_lerobot" / "farm_uf850_bottle" / "meta" / "tasks.jsonl"


# ── helpers ─────────────────────────────────────────────────────────────────


def load_known_tasks() -> list[str]:
    if not TASKS_PATH.is_file():
        return []
    out: list[str] = []
    with TASKS_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = row.get("task")
            if isinstance(t, str):
                out.append(t)
    return out


def decode_jpeg(blob: bytes, *, resize: tuple[int, int] | None) -> np.ndarray:
    """JPEG bytes → uint8 HxWx3 RGB. Aspect-preserving resize-with-pad
    (matches openpi's training preprocessing). Plain ``img.resize()``
    SQUISHES the 4:3 RealSense image to 1:1 — the model has never seen
    distorted images and grasp accuracy drops noticeably. ``resize_with_pad``
    instead scales-to-fit and pads the short side with black, which is
    what the dataset was trained against.
    """
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    if resize is None or img.size == resize:
        return np.asarray(img, dtype=np.uint8)
    # Replicates openpi/shared/image_tools.py resize_with_pad. Target
    # (width, height) is symmetric for 224×224.
    target_w, target_h = resize
    cur_w, cur_h = img.size
    ratio = max(cur_w / target_w, cur_h / target_h)
    new_w = int(round(cur_w / ratio))
    new_h = int(round(cur_h / ratio))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


# ── daemon client ───────────────────────────────────────────────────────────


@dataclass
class FarmObs:
    state: np.ndarray            # (7,) float32 — 6 joints (rad) + gripper (0..1)
    base_rgb: np.ndarray         # (H, W, 3) uint8
    wrist_rgb: np.ndarray        # (H, W, 3) uint8
    drive_real_arm: bool
    estopped: bool
    raw_snapshot: dict[str, Any]


class FarmDaemonClient:
    def __init__(self, base_url: str, *, image_shape: tuple[int, int]) -> None:
        self._base = base_url.rstrip("/")
        self._image_shape = image_shape
        self._session = requests.Session()
        # Last successful decode per camera, kept as a fallback for the
        # next request when the daemon's grabber transiently 503s. A
        # stale frame is far preferable to crashing the eval loop, and
        # the policy only sees one duplicate frame in practice.
        self._last_good: dict[str, np.ndarray] = {}

    def world(self) -> dict[str, Any]:
        r = self._session.get(f"{self._base}/v1/world", timeout=2.0)
        r.raise_for_status()
        return r.json()

    def camera(self, name: str) -> bytes:
        # /v1/cameras returns 503 whenever the RealSense grabber's
        # latest_jpeg cache is empty — happens on cold start and
        # occasionally between frames. Retry with backoff for up to
        # ~1.5 s before giving up.
        delays = (0.05, 0.1, 0.2, 0.4, 0.8)
        last_code = 0
        for delay in delays:
            r = self._session.get(
                f"{self._base}/v1/cameras/{name}.jpg", timeout=2.0
            )
            if r.status_code == 200:
                return r.content
            last_code = r.status_code
            time.sleep(delay)
        # One last attempt without sleep so the final read is fresh.
        r = self._session.get(
            f"{self._base}/v1/cameras/{name}.jpg", timeout=2.0
        )
        if r.status_code == 200:
            return r.content
        raise RuntimeError(
            f"camera {name!r} returned {last_code} on every attempt — "
            "is the RealSense grabber alive? Check /v1/hud."
        )

    def _fetch_or_reuse(self, name: str) -> np.ndarray:
        try:
            blob = self.camera(name)
            img = decode_jpeg(blob, resize=self._image_shape)
            self._last_good[name] = img
            return img
        except RuntimeError as exc:
            cached = self._last_good.get(name)
            if cached is not None:
                print(
                    f"[eval] WARN camera {name!r} fetch failed ({exc}); "
                    "reusing last good frame.",
                    file=sys.stderr,
                )
                return cached
            print(
                f"[eval] WARN camera {name!r} unavailable on first fetch ({exc}); "
                "using a zero frame so the loop can proceed. Policy output may "
                "be meaningless until the camera comes back.",
                file=sys.stderr,
            )
            h, w = self._image_shape
            zero = np.zeros((h, w, 3), dtype=np.uint8)
            self._last_good[name] = zero
            return zero

    def observation(self, *, task: str) -> FarmObs:
        snap = self.world()
        joints = snap.get("joints") or []
        if len(joints) < 6:
            raise RuntimeError(f"snapshot has only {len(joints)} joints; expected 6")
        grip = snap.get("gripper_pos")
        if grip is None or not isinstance(grip, (int, float)):
            grip = 0.0
        state = np.array(
            [float(joints[i]) for i in range(6)] + [float(grip)],
            dtype=np.float32,
        )
        base = self._fetch_or_reuse("base")
        wrist = self._fetch_or_reuse("wrist")
        _ = task  # task is the policy's prompt, not used here — kept for symmetry
        return FarmObs(
            state=state,
            base_rgb=base,
            wrist_rgb=wrist,
            drive_real_arm=bool(snap.get("drive_real_arm", False)),
            estopped=bool(snap.get("estopped", False)),
            raw_snapshot=snap,
        )

    def post_joints(
        self,
        joints_rad: list[float],
        *,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"joints": [float(j) for j in joints_rad]}
        if gripper is not None:
            body["gripper"] = float(gripper)
        r = self._session.post(
            f"{self._base}/v1/teleop/joints", json=body, timeout=2.0
        )
        if r.status_code >= 400:
            raise RuntimeError(f"POST /v1/teleop/joints → {r.status_code}: {r.text}")
        return r.json()

    def set_motion_scale(self, scale: float) -> dict[str, Any]:
        r = self._session.post(
            f"{self._base}/v1/teleop/motion_scale",
            json={"scale": float(scale)},
            timeout=2.0,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"POST /v1/teleop/motion_scale → {r.status_code}: {r.text}"
            )
        return r.json()

    def estop(self) -> dict[str, Any]:
        r = self._session.post(f"{self._base}/v1/teleop/estop", timeout=2.0)
        r.raise_for_status()
        return r.json()

    def policy_prompt(self) -> str:
        """GET the dashboard-set prompt. Empty string if the daemon
        doesn't have one (or the endpoint is missing on an older
        daemon). Caller decides what to fall back to."""
        try:
            r = self._session.get(f"{self._base}/v1/policy/prompt", timeout=1.5)
            if r.status_code != 200:
                return ""
            body = r.json()
            return str(body.get("prompt", ""))
        except Exception:
            return ""

    def heartbeat(self, **fields: Any) -> None:
        """Best-effort POST a liveness/status ping the dashboard surfaces.
        Failures are swallowed — heartbeats are advisory, not load-bearing."""
        try:
            self._session.post(
                f"{self._base}/v1/policy/heartbeat",
                json=fields, timeout=0.8,
            )
        except Exception:
            pass


# ── policy abstraction ──────────────────────────────────────────────────────


class Policy:
    action_dim_reported: int = 0
    action_horizon_reported: int = 0

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        """Return action chunk shape (H, action_dim)."""
        raise NotImplementedError

    def close(self) -> None: ...


class StubPolicy(Policy):
    """Returns zero deltas + gripper held at 0 (open). Use to validate
    daemon ↔ client plumbing without spinning up a GPU."""

    def __init__(self, *, horizon: int = 10, action_dim: int = 32) -> None:
        self._h = horizon
        self._d = action_dim
        self.action_dim_reported = action_dim
        self.action_horizon_reported = horizon

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        _ = obs
        return np.zeros((self._h, self._d), dtype=np.float32)


# openpi-client's bespoke msgpack-numpy variant (bytes keys, no pickle
# fallback). Ported from
# openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py so the
# eval client doesn't need the openpi-client package installed locally.
def _pack_ndarray(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_ndarray(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class WebSocketPolicy(Policy):
    """Talks to ``openpi serve_policy.py`` via its WebSocket protocol.

    Protocol (matches ``openpi-client/websocket_client_policy.py``):

    * Connect with ``compression=None, max_size=None``
    * Server sends a msgpack-packed metadata dict on connect
    * Client sends msgpack-packed obs dict; server replies with
      msgpack-packed result dict OR a *string* error message
    * Encoding uses openpi's custom np-array packer (bytes keys, no
      pickle fallback) — incompatible with the PyPI ``msgpack-numpy``.

    The first ``infer`` call may take seconds while the server
    JIT-compiles its sampling graph. After that, ~50–100 ms on H100.
    """

    def __init__(self, url: str, *, connect_timeout: float = 10.0) -> None:
        try:
            import msgpack  # noqa: F401  (sanity)
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "WebSocketPolicy needs msgpack + websockets. "
                "Install with: pip install msgpack websockets"
            ) from exc
        import msgpack as _msgpack
        self._connect = connect
        self._url = url
        self._connect_timeout = connect_timeout
        self._ws = None
        # Bind packer/unpacker variants matching the openpi-client wire format.
        self._packer = _msgpack.Packer(default=_pack_ndarray)
        self._unpackb = lambda raw: _msgpack.unpackb(raw, object_hook=_unpack_ndarray)
        self._connect_ws()

    def _connect_ws(self) -> None:
        # ``compression=None`` and ``max_size=None`` mirror openpi-client.
        # The default permessage-deflate negotiation corrupts msgpack
        # payloads in some setups, and the default 1 MB max_size truncates
        # 224×224 RGB observations.
        self._ws = self._connect(
            self._url,
            open_timeout=self._connect_timeout,
            compression=None,
            max_size=None,
        )
        try:
            metadata_raw = self._ws.recv()
            if isinstance(metadata_raw, str):
                raise RuntimeError(
                    f"server sent text metadata frame (expected binary): {metadata_raw!r}"
                )
            metadata = self._unpackb(metadata_raw)
            if isinstance(metadata, dict):
                self.action_dim_reported = int(metadata.get("action_dim", 0))
                self.action_horizon_reported = int(metadata.get("action_horizon", 0))
        except Exception:
            # Older servers may not send metadata; that's fine.
            pass

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        if self._ws is None:
            self._connect_ws()
        payload = self._packer.pack(obs)
        self._ws.send(payload)
        reply = self._ws.recv()
        # openpi's server returns a TEXT frame containing the traceback
        # when an inference error happens (e.g. a transform throws). Surface
        # that intact so we don't try to msgpack-decode an error string.
        if isinstance(reply, str):
            raise RuntimeError(f"server-side inference error:\n{reply}")
        result = self._unpackb(reply)
        if isinstance(result, dict) and "actions" in result:
            actions = np.asarray(result["actions"])
        else:
            actions = np.asarray(result)
        if actions.ndim == 1:
            actions = actions[None, :]
        return actions.astype(np.float32, copy=False)

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        self._ws = None


# ── main eval loop ──────────────────────────────────────────────────────────


@dataclass
class LoopConfig:
    task: str
    rate_hz: float
    steps_per_chunk: int
    motion_scale: float
    max_steps: int
    action_mode: str       # "delta" or "absolute"
    no_gripper: bool
    require_drive_real_arm: bool
    dry_run: bool
    rtc: bool = True       # Real-Time Chunking (server-side guided seam blend)
    stream_hz: float = 0.0  # steady interpolated POST rate (0 = proven per-action loop; e.g. 100 to enable)
    deltas_seen: list[np.ndarray] = field(default_factory=list)


def _make_policy_obs(
    obs: FarmObs, *, prompt: str, rtc: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the obs dict the openpi server expects.

    The server runs ``LiberoInputs`` (reused by ``LeRobotFarmDataConfig``)
    which reads the openpi-internal keys ``observation/image``,
    ``observation/wrist_image``, ``observation/state``, and ``prompt``.
    The RepackTransform that maps LeRobot column names → these internal
    keys runs at *training* time only; inference inputs must already be
    in the internal format.

    ``rtc`` (optional) carries Real-Time Chunking control fields the patched
    server pops off before the input transform: ``rtc_reset`` (drop server
    state at episode start), ``rtc_offset`` (steps advanced since the obs that
    produced the previous chunk), ``rtc_delay`` (inference delay in steps →
    hard-frozen prefix length). With RTC the server guides the next chunk to
    join continuously with the previous one over their overlap.
    """
    out: dict[str, Any] = {
        "observation/image":       obs.base_rgb,    # (H, W, 3) uint8
        "observation/wrist_image": obs.wrist_rgb,   # (H, W, 3) uint8
        "observation/state":       obs.state,       # (7,) float32
        "prompt":                  prompt,
    }
    if rtc:
        out.update(rtc)
    return out


def _format_action_row(action: np.ndarray) -> str:
    j = ", ".join(f"{x:+.4f}" for x in action[:6])
    g = float(action[6]) if action.shape[0] > 6 else float("nan")
    return f"joints[Δrad]=[{j}]  gripper={g:+.3f}"


def _apply_delta_chunk(
    *,
    base_state: np.ndarray,
    action_chunk: np.ndarray,
    motion_scale: float,
    action_mode: str,
    steps_per_chunk: int,
) -> list[tuple[np.ndarray, float]]:
    """Project a policy action chunk into per-step (target_joints, gripper).

    * ``delta`` mode: first 6 dims are joint deltas. We cumulatively
      add them to ``base_state[:6]`` scaled by ``motion_scale``.
    * ``absolute`` mode: first 6 dims are absolute joint targets.
    * dim 6 is always treated as absolute gripper in [0, 1]
      regardless of action_mode (matches the trained policy spec).
    """
    out: list[tuple[np.ndarray, float]] = []
    cum = np.array(base_state[:6], dtype=np.float64, copy=True)
    for i in range(min(steps_per_chunk, action_chunk.shape[0])):
        a = action_chunk[i]
        if action_mode == "delta":
            cum = cum + motion_scale * a[:6].astype(np.float64)
            target = cum.copy()
        else:  # absolute
            target = a[:6].astype(np.float64)
        gripper = float(np.clip(a[6], 0.0, 1.0)) if a.shape[0] > 6 else 0.0
        out.append((target, gripper))
    return out


def run_stream_loop(
    *,
    daemon: FarmDaemonClient,
    policy: Policy,
    cfg: LoopConfig,
    policy_url: str,
) -> None:
    """Steady high-rate streaming loop.

    The policy emits a sparse chunk (H absolute joint waypoints spaced at
    ``1/cfg.rate_hz``). Committing those directly leaves the daemon's 250 Hz
    servo tracker chasing sparse, irregularly-timed targets (and the legacy loop
    *skipped* stale ones → visible jumps). Here we resample the chunk into a
    STEADY ``cfg.stream_hz`` (default 100 Hz) stream by linear interpolation
    between waypoints and POST at that fixed rate, so the daemon always has a
    dense, fresh, uniformly-timed target → smooth continuous motion.

    Latency is absorbed implicitly. Each chunk is anchored in wall-clock at its
    own ``obs_time`` (knot 0 = the measured joint state at capture, knots 1..H =
    the predicted action targets at ``k/cfg.rate_hz``). We always render the
    trajectory at "now", so when a fresher (RTC-guided) chunk lands we just
    re-anchor and keep interpolating forward from wherever real time has reached
    — no skip, no jump. RTC makes the new chunk's overlap continue the previous
    one, so the re-anchor is seamless. Inference runs continuously pipelined
    (one always in flight) in a background thread.

    POSTs use a dedicated requests session so the 100 Hz POST stream never races
    the background obs-fetch session.
    """
    out_hz = float(cfg.stream_hz)
    out_dt = 1.0 / out_hz
    wp_dt = 1.0 / max(0.5, cfg.rate_hz)   # seconds between policy waypoints
    print(
        f"[eval] STREAMING · task={cfg.task!r} · chunk_rate={cfg.rate_hz}Hz · "
        f"output={out_hz:.0f}Hz (interpolated) · motion_scale={cfg.motion_scale} · "
        f"action_mode={cfg.action_mode} · rtc={'ON' if cfg.rtc else 'off'} · dry_run={cfg.dry_run}"
    )

    rtc_state: dict[str, Any] = {
        "prev_obs_time": None,
        "delay_steps": max(1, int(round(0.18 * cfg.rate_hz))),
        "horizon": 10,
    }

    def _build_rtc(obs_wall_t: float, *, reset: bool) -> dict[str, Any] | None:
        if not cfg.rtc:
            return None
        if reset or rtc_state["prev_obs_time"] is None:
            return {"rtc_reset": True}
        h = int(rtc_state["horizon"])
        offset = int(round((obs_wall_t - rtc_state["prev_obs_time"]) * cfg.rate_hz))
        offset = max(1, min(h - 1, offset))
        overlap = h - offset
        delay = max(0, min(overlap, int(rtc_state["delay_steps"])))
        return {"rtc_offset": offset, "rtc_delay": delay}

    def _infer(*, reset: bool) -> dict[str, Any]:
        obs_wall_t = time.perf_counter()
        try:
            o = daemon.observation(task=cfg.task)
        except Exception as exc:
            return {"error": f"obs fetch failed: {exc}"}
        if o.estopped:
            return {"estopped": True, "obs": o}
        prompt = daemon.policy_prompt() or cfg.task
        rtc = _build_rtc(obs_wall_t, reset=reset)
        try:
            t0 = time.perf_counter()
            ac = policy.infer(_make_policy_obs(o, prompt=prompt, rtc=rtc))
            return {
                "ok": True, "obs": o, "obs_time": obs_wall_t, "prompt": prompt,
                "action_chunk": ac, "infer_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as exc:
            return {"ok": False, "error": f"infer failed: {exc}", "obs": o, "obs_time": obs_wall_t}

    def _knots(o: FarmObs, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interp knots: knot 0 = measured state at obs capture, knots 1..H =
        the chunk's joint targets (cumulative in delta mode, absolute else)."""
        h = int(chunk.shape[0])
        pos = np.zeros((h + 1, 6), dtype=np.float64)
        grip = np.zeros(h + 1, dtype=np.float64)
        pos[0] = np.asarray(o.state[:6], dtype=np.float64)
        grip[0] = float(o.state[6]) if o.state.shape[0] > 6 else 0.0
        for k in range(h):
            a = chunk[k]
            if cfg.action_mode == "delta":
                pos[k + 1] = pos[k] + cfg.motion_scale * a[:6].astype(np.float64)
            else:
                pos[k + 1] = a[:6].astype(np.float64)
            grip[k + 1] = float(np.clip(a[6], 0.0, 1.0)) if a.shape[0] > 6 else grip[k]
        return pos, grip

    def _interp(pos: np.ndarray, grip: np.ndarray, tau: float) -> tuple[np.ndarray, float]:
        h = pos.shape[0] - 1
        span = h * wp_dt
        if tau <= 0.0:
            return pos[0], float(grip[0])
        if tau >= span:
            return pos[h], float(grip[h])
        kf = tau / wp_dt
        k = int(kf)
        f = kf - k
        return pos[k] * (1.0 - f) + pos[k + 1] * f, float(grip[k] * (1.0 - f) + grip[k + 1] * f)

    # Dedicated POST session — keeps the 100 Hz command stream off the same
    # requests.Session the background obs-fetch uses (not concurrency-safe).
    post_session = requests.Session()
    post_url = f"{daemon._base}/v1/teleop/joints"

    def _post(joints: np.ndarray, gripper: float | None) -> dict[str, Any]:
        body: dict[str, Any] = {"joints": [float(j) for j in joints]}
        if gripper is not None:
            body["gripper"] = float(gripper)
        r = post_session.post(post_url, json=body, timeout=1.0)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    daemon.heartbeat(policy_url=policy_url, policy_ok=None, task_prompt=cfg.task,
                     drive_real_arm=False, dry_run=cfg.dry_run, note="warming up")

    # Bootstrap (synchronous): need the first chunk before any motion.
    cur = _infer(reset=True)
    if cur.get("estopped"):
        print("[eval] daemon e-stopped at start — aborting", file=sys.stderr)
        return
    if not cur.get("ok"):
        print(f"[eval] first inference FAILED: {cur.get('error')}", file=sys.stderr)
        return
    obs = cur["obs"]
    pos, grip = _knots(obs, cur["action_chunk"])
    H = pos.shape[0] - 1
    chunk_t0 = time.perf_counter()         # render the bootstrap chunk starting now
    rtc_state.update(prev_obs_time=chunk_t0, horizon=H)
    if cur.get("infer_ms"):
        rtc_state["delay_steps"] = max(1, int(round(cur["infer_ms"] / 1000 * cfg.rate_hz)))
    live_prompt = cur["prompt"]
    n_chunks = 1
    cfg.deltas_seen.append(cur["action_chunk"][:, :7].copy())
    print(f"[chunk {n_chunks:3d}] {tuple(cur['action_chunk'].shape)} in {cur.get('infer_ms', 0):.0f}ms")
    daemon.heartbeat(policy_url=policy_url, policy_ok=True, task_prompt=live_prompt,
                     drive_real_arm=obs.drive_real_arm, dry_run=cfg.dry_run,
                     last_chunk_ms=cur.get("infer_ms"), last_action_idx=0)

    # Continuous pipelined inference: always keep one in flight.
    bg_out: dict[str, dict[str, Any]] = {"r": {}}
    bg: threading.Thread | None = None
    consecutive_fail = 0

    def _spawn() -> None:
        nonlocal bg
        bg_out["r"] = {}
        bg = threading.Thread(target=lambda: bg_out["r"].update(_infer(reset=False)),
                              daemon=True, name="bg-infer")
        bg.start()

    _spawn()

    start_wall = time.perf_counter()
    budget_s = cfg.max_steps / max(1.0, cfg.rate_hz)
    posts = 0
    next_tick = time.perf_counter()
    last_log = 0.0
    now = start_wall
    while True:
        now = time.perf_counter()
        if now - start_wall >= budget_s:
            break
        tau = now - chunk_t0
        joints, gripper = _interp(pos, grip, tau)
        gripper_send = None if cfg.no_gripper else gripper

        if cfg.dry_run:
            if now - last_log >= 0.5:
                last_log = now
                line = (f"  [stream] post#{posts} tau={tau * 1000:4.0f}ms "
                        f"j=[{', '.join(f'{x:+.3f}' for x in joints)}]")
                if gripper_send is not None:
                    line += f" grip={gripper_send:+.2f}"
                print(line)
        else:
            try:
                resp = _post(joints, gripper_send)
                if isinstance(resp, dict) and resp.get("drive_real_arm") is False and obs.drive_real_arm:
                    print("  drive_real_arm turned OFF mid-run; real arm stopped.", file=sys.stderr)
            except Exception as exc:
                msg = str(exc)
                if "estop" in msg.lower() or "409" in msg:
                    print(f"  POST refused (likely e-stop): {msg}", file=sys.stderr)
                    daemon.heartbeat(policy_url=policy_url, policy_ok=True, task_prompt=live_prompt,
                                     drive_real_arm=obs.drive_real_arm, dry_run=cfg.dry_run,
                                     last_action_idx=posts, note="estopped")
                    return
                print(f"  POST failed: {msg}", file=sys.stderr)
                return
        posts += 1

        # Adopt a freshly-inferred chunk; re-anchor at its obs_time and keep
        # interpolating forward (latency lands us partway in → no jump).
        if bg is not None and not bg.is_alive():
            r = bg_out["r"]
            bg = None
            if r.get("estopped"):
                print("[eval] daemon e-stopped — aborting", file=sys.stderr)
                return
            if r.get("ok"):
                consecutive_fail = 0
                obs = r["obs"]
                pos, grip = _knots(obs, r["action_chunk"])
                H = pos.shape[0] - 1
                chunk_t0 = r["obs_time"]
                rtc_state.update(prev_obs_time=chunk_t0, horizon=H)
                if r.get("infer_ms"):
                    rtc_state["delay_steps"] = max(1, int(round(r["infer_ms"] / 1000 * cfg.rate_hz)))
                live_prompt = r["prompt"]
                n_chunks += 1
                cfg.deltas_seen.append(r["action_chunk"][:, :7].copy())
                if n_chunks % 10 == 0:
                    daemon.heartbeat(policy_url=policy_url, policy_ok=True, task_prompt=live_prompt,
                                     drive_real_arm=obs.drive_real_arm, dry_run=cfg.dry_run,
                                     last_chunk_ms=r.get("infer_ms"), last_action_idx=posts)
            else:
                consecutive_fail += 1
                print(f"[eval] infer failed ({consecutive_fail}): {r.get('error')}", file=sys.stderr)
                if consecutive_fail >= 5:
                    print("[eval] too many inference failures — aborting", file=sys.stderr)
                    return
            _spawn()   # keep one inference always in flight

        # Steady fixed-rate tick.
        next_tick += out_dt
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()   # fell behind — resync, don't burst

    elapsed = max(1e-6, now - start_wall)
    print(f"[eval] streaming done · {posts} posts over {elapsed:.1f}s "
          f"(~{posts / elapsed:.0f} Hz effective) · {n_chunks} chunks")
    daemon.heartbeat(policy_url=policy_url, policy_ok=True, task_prompt=live_prompt,
                     dry_run=cfg.dry_run, last_action_idx=posts, note="done · streaming")


def run_loop(
    *,
    daemon: FarmDaemonClient,
    policy: Policy,
    cfg: LoopConfig,
    policy_url: str,
    log_every: int = 1,
) -> None:
    """Pipelined eval loop with temporal alignment.

    The two prior attempts each had one nasty failure mode:

    * Synchronous: 180 ms inference pause between every chunk. Breaks
      the trajectory dynamics — training was continuous teleop, so the
      policy never learned to handle start-stop-start motion. End
      effector overshoots at the pause-resume transitions and misses
      small grasp targets like the bottle.

    * Pipelined (naive): obs captured mid-chunk N; by the time chunk
      N+1 actually executed, its first few actions targeted trajectory
      time that already passed → arm "rubber-banded" backward.

    Fix: each predicted action is treated as a TIMED WAYPOINT. The
    policy predicts action[i] for trajectory time ``obs_time + (i+1)·period``.
    When chunk N+1's inference completes, we skip every action whose
    target time is already in the past (because chunk N's commits got
    us there), and commit the rest at their intended wall-clock times.
    Result: continuous 30 Hz motion with no pauses AND no rubber-band.
    """
    period = 1.0 / max(0.5, cfg.rate_hz)
    print(
        f"[eval] starting · task='{cfg.task}' · rate={cfg.rate_hz}Hz · "
        f"steps_per_chunk={cfg.steps_per_chunk} · motion_scale={cfg.motion_scale} · "
        f"action_mode={cfg.action_mode} · dry_run={cfg.dry_run} · "
        f"rtc={'ON' if cfg.rtc else 'off'} · pipelined=True (timed)"
    )
    last_prompt: str | None = None

    # Initial heartbeat — surfaces the eval client in the dashboard
    # before the first inference completes (which on a cold policy
    # server can be 10–30 s while JAX compiles).
    daemon.heartbeat(
        policy_url=policy_url, policy_ok=None,
        task_prompt=cfg.task, drive_real_arm=False, dry_run=cfg.dry_run,
        note="warming up",
    )

    # Real-Time Chunking state shared with the background inference thread.
    #   prev_obs_time : perf_counter() anchor of the chunk currently executing
    #   delay_steps   : rolling inference latency expressed in action steps
    #                   (= size of the server-side hard-frozen prefix)
    #   horizon       : chunk length H (filled in from the first chunk)
    rtc_state: dict[str, Any] = {
        "prev_obs_time": None,
        "delay_steps": max(1, int(round(0.18 * cfg.rate_hz))),
        "horizon": DEFAULT_STEPS_PER_CHUNK * 2,
    }

    def _build_rtc(obs_wall_t: float, *, reset: bool) -> dict[str, Any] | None:
        """Compute the RTC control fields for one inference. Returns None when
        RTC is disabled. ``rtc_offset`` is how many action-steps elapsed between
        the executing chunk's obs and this one (so the server can align the two
        chunks); ``rtc_delay`` is the inference delay in steps (frozen prefix)."""
        if not cfg.rtc:
            return None
        if reset or rtc_state["prev_obs_time"] is None:
            return {"rtc_reset": True}
        h = int(rtc_state["horizon"])
        offset = int(round((obs_wall_t - rtc_state["prev_obs_time"]) * cfg.rate_hz))
        offset = max(1, min(h - 1, offset))
        overlap = h - offset
        delay = max(0, min(overlap, int(rtc_state["delay_steps"])))
        return {"rtc_offset": offset, "rtc_delay": delay}

    def _fetch_and_infer(*, reset: bool = False) -> dict[str, Any]:
        """One (obs fetch → infer) pass. Records ``obs_time`` so the
        caller can compute the wall-clock anchor for each predicted
        action in the returned chunk."""
        out: dict[str, Any] = {}
        obs_wall_t = time.perf_counter()
        try:
            o = daemon.observation(task=cfg.task)
        except Exception as exc:
            out["error"] = f"obs fetch failed: {exc}"
            return out
        if o.estopped:
            out["estopped"] = True
            out["obs"] = o
            return out
        prompt = daemon.policy_prompt() or cfg.task
        rtc = _build_rtc(obs_wall_t, reset=reset)
        try:
            t0 = time.perf_counter()
            ac = policy.infer(_make_policy_obs(o, prompt=prompt, rtc=rtc))
            out["infer_ms"] = (time.perf_counter() - t0) * 1000
            out["obs"] = o
            out["obs_time"] = obs_wall_t
            out["prompt"] = prompt
            out["action_chunk"] = ac
            out["ok"] = True
        except Exception as exc:
            out["ok"] = False
            out["error"] = f"infer failed: {exc}"
            out["obs"] = o
            out["obs_time"] = obs_wall_t
            out["prompt"] = prompt
        return out

    # Bootstrap synchronously: we need the first chunk before any motion.
    # reset=True drops any RTC state left on the server from a prior run.
    current = _fetch_and_infer(reset=True)
    n_chunks = 1
    if current.get("estopped"):
        print("[eval] daemon is e-stopped at start — aborting", file=sys.stderr)
        daemon.heartbeat(
            policy_url=policy_url, policy_ok=None,
            task_prompt=cfg.task, drive_real_arm=False, dry_run=cfg.dry_run,
            note="estopped at start",
        )
        return
    if not current.get("ok"):
        print(f"[eval] first inference FAILED: {current.get('error')}", file=sys.stderr)
        daemon.heartbeat(
            policy_url=policy_url, policy_ok=False,
            task_prompt=cfg.task, drive_real_arm=False, dry_run=cfg.dry_run,
            note=current.get("error", "first infer failed"),
        )
        return

    def _log_chunk(idx: int, result: dict[str, Any]) -> None:
        ac = result["action_chunk"]
        infer_ms = float(result.get("infer_ms", 0.0))
        first = ac[0] if ac.shape[0] > 0 else None
        print(
            f"[chunk {idx:3d}] inferred shape={ac.shape} "
            f"in {infer_ms:5.1f}ms · "
            f"first={_format_action_row(first) if first is not None else 'EMPTY'}"
        )
        cfg.deltas_seen.append(ac[:, :7].copy())
        mags = np.linalg.norm(ac[:, :6], axis=1)
        print(
            f"             |joint|         min={mags.min():.4f} "
            f"max={mags.max():.4f} mean={mags.mean():.4f}  "
            f"gripper·chunk min={float(ac[:,6].min()):.3f} "
            f"max={float(ac[:,6].max()):.3f}"
        )

    _log_chunk(n_chunks, current)
    obs = current["obs"]
    live_prompt = current["prompt"]
    action_chunk = current["action_chunk"]
    daemon.heartbeat(
        policy_url=policy_url, policy_ok=True,
        task_prompt=live_prompt, drive_real_arm=obs.drive_real_arm,
        dry_run=cfg.dry_run, last_chunk_ms=current.get("infer_ms"),
        last_action_idx=0,
    )

    step = 0
    # Bootstrap reset: the bootstrap chunk's obs was captured at
    # startup, before whatever startup overhead and JAX warmup the
    # rest of the script took (potentially 10-30 s on a cold server).
    # Treat it as "freshly captured now" so the first chunk uses all
    # 10 of its predictions instead of skipping 9 as stale. Subsequent
    # chunks use their actual obs_time (set from bg_thread results).
    obs_time = time.perf_counter()
    # Seed RTC state from the bootstrap chunk so the first background inference
    # can be aligned against it.
    rtc_state["horizon"] = int(action_chunk.shape[0])
    rtc_state["prev_obs_time"] = obs_time
    if current.get("infer_ms"):
        rtc_state["delay_steps"] = max(1, int(round(current["infer_ms"] / 1000 * cfg.rate_hz)))
    next_result: dict[str, Any] = {}
    next_result_holder: dict[str, dict[str, Any]] = {"r": {}}
    bg_thread: threading.Thread | None = None

    def _spawn_bg() -> None:
        nonlocal bg_thread
        next_result_holder["r"] = {}
        bg_thread = threading.Thread(
            target=lambda: next_result_holder["r"].update(_fetch_and_infer()),
            daemon=True, name="bg-infer",
        )
        bg_thread.start()

    while step < cfg.max_steps:
        plan = _apply_delta_chunk(
            base_state=obs.state,
            action_chunk=action_chunk,
            motion_scale=cfg.motion_scale,
            action_mode=cfg.action_mode,
            steps_per_chunk=cfg.steps_per_chunk,
        )

        if last_prompt is None and live_prompt != cfg.task:
            print(
                f"[eval] dashboard prompt active: {live_prompt!r} "
                "(overrides --task)"
            )
        elif last_prompt is not None and live_prompt != last_prompt:
            print(f"[eval] dashboard prompt changed: {live_prompt!r}")
        last_prompt = live_prompt

        # Temporal alignment: each action[i] was predicted for wall
        # time ``obs_time + (i+1)·period``. Skip any whose target time
        # is already in the past (chunk N+1's first few actions land
        # in the past because the bg infer took ~180 ms).
        elapsed_since_obs = time.perf_counter() - obs_time
        skip = max(0, min(len(plan) - 1, int(round(elapsed_since_obs * cfg.rate_hz))))
        if skip > 0:
            print(
                f"  [align] skipping {skip} stale action(s) "
                f"(obs was {elapsed_since_obs*1000:.0f} ms old)"
            )
        # Spawn the next chunk's inference roughly halfway through the
        # *remaining* actions so the ~180 ms infer finishes near the
        # chunk's last commit, keeping motion continuous.
        remaining = len(plan) - skip
        spawn_after_step = skip + max(0, remaining // 2)

        for i in range(skip, len(plan)):
            if step >= cfg.max_steps:
                break

            if bg_thread is None and i >= spawn_after_step:
                _spawn_bg()

            # Timed waypoint commit — sleep until this action's
            # absolute wall time.
            target_wall = obs_time + (i + 1) * period
            now = time.perf_counter()
            wait_for = target_wall - now
            if wait_for > 0:
                time.sleep(wait_for)

            joints, gripper = plan[i]
            gripper_send = None if cfg.no_gripper else gripper
            label = f"  step {step:4d}.{i}"
            if cfg.dry_run:
                line = (
                    f"{label} DRY  target_joints=["
                    + ", ".join(f"{j:+.4f}" for j in joints.tolist())
                    + "]"
                )
                if gripper_send is not None:
                    line += f"  gripper={gripper_send:+.3f}"
                print(line)
            else:
                try:
                    resp = daemon.post_joints(
                        list(joints.tolist()), gripper=gripper_send
                    )
                except Exception as exc:
                    msg = str(exc)
                    if "estopped" in msg.lower() or "409" in msg:
                        print(
                            f"{label} POST refused by daemon "
                            f"(likely e-stop): {msg}",
                            file=sys.stderr,
                        )
                    else:
                        print(f"{label} POST failed: {msg}", file=sys.stderr)
                    daemon.heartbeat(
                        policy_url=policy_url, policy_ok=True,
                        task_prompt=live_prompt,
                        drive_real_arm=obs.drive_real_arm,
                        dry_run=cfg.dry_run, last_action_idx=step,
                        note=f"post failed: {msg}",
                    )
                    return
                if (
                    isinstance(resp, dict)
                    and resp.get("drive_real_arm") is False
                    and obs.drive_real_arm
                ):
                    print(
                        f"{label} drive_real_arm was turned OFF mid-chunk; "
                        "ghost still tracks but real arm has stopped.",
                        file=sys.stderr,
                    )
                if step % log_every == 0:
                    line = (
                        f"{label} LIVE target_joints=["
                        + ", ".join(f"{j:+.4f}" for j in joints.tolist())
                        + "]"
                    )
                    if gripper_send is not None:
                        line += f"  gripper={gripper_send:+.3f}"
                    print(line)
            step += 1

        if step >= cfg.max_steps:
            break

        # Pick up the pipelined next chunk. If bg never spawned (e.g.
        # the entire chunk was skipped as stale, which only happens on
        # severe lag), fall back to synchronous.
        if bg_thread is not None:
            bg_thread.join()
            bg_thread = None
            next_result = next_result_holder["r"]
        else:
            next_result = _fetch_and_infer()

        if next_result.get("estopped"):
            print("[eval] daemon is e-stopped — aborting loop", file=sys.stderr)
            daemon.heartbeat(
                policy_url=policy_url, policy_ok=True,
                task_prompt=live_prompt,
                drive_real_arm=next_result["obs"].drive_real_arm,
                dry_run=cfg.dry_run, last_action_idx=step, note="estopped",
            )
            return
        n_chunks += 1
        if not next_result.get("ok"):
            err = next_result.get("error", "unknown")
            print(
                f"[chunk {n_chunks:3d}] infer FAILED: {err}",
                file=sys.stderr,
            )
            daemon.heartbeat(
                policy_url=policy_url, policy_ok=False,
                task_prompt=live_prompt, drive_real_arm=obs.drive_real_arm,
                dry_run=cfg.dry_run, last_action_idx=step,
                note=f"infer failed: {err}",
            )
            # Pause briefly then retry synchronously to give the server
            # a chance to recover (e.g. transient WebSocket hiccup).
            time.sleep(0.5)
            current = _fetch_and_infer()
            if not current.get("ok"):
                print("[eval] retry also failed; aborting", file=sys.stderr)
                return
            next_result = current

        _log_chunk(n_chunks, next_result)
        obs = next_result["obs"]
        obs_time = next_result["obs_time"]
        # Advance RTC state: this chunk is now the one executing, so it becomes
        # the alignment anchor for the next background inference.
        rtc_state["prev_obs_time"] = obs_time
        rtc_state["horizon"] = int(next_result["action_chunk"].shape[0])
        if next_result.get("infer_ms"):
            rtc_state["delay_steps"] = max(
                1, int(round(next_result["infer_ms"] / 1000 * cfg.rate_hz))
            )
        live_prompt = next_result["prompt"]
        action_chunk = next_result["action_chunk"]
        daemon.heartbeat(
            policy_url=policy_url, policy_ok=True,
            task_prompt=live_prompt, drive_real_arm=obs.drive_real_arm,
            dry_run=cfg.dry_run, last_chunk_ms=next_result.get("infer_ms"),
            last_action_idx=step,
        )

    print(f"[eval] hit max_steps={cfg.max_steps}; stopping cleanly.")
    daemon.heartbeat(
        policy_url=policy_url, policy_ok=True,
        task_prompt=last_prompt or cfg.task, dry_run=cfg.dry_run,
        last_action_idx=step, note=f"done · max_steps={cfg.max_steps}",
    )


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_pi05",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task", default=DEFAULT_TASK,
        help=f"language prompt (default: {DEFAULT_TASK!r}). Use --list-tasks to print the trained tasks.",
    )
    p.add_argument(
        "--list-tasks", action="store_true",
        help="print the trained tasks and exit",
    )
    p.add_argument(
        "--daemon-url", default=DEFAULT_DAEMON_URL,
        help=f"FARM daemon base URL (default: {DEFAULT_DAEMON_URL})",
    )
    p.add_argument(
        "--policy-url", default=DEFAULT_POLICY_URL,
        help=f"openpi serve_policy.py WebSocket URL (default: {DEFAULT_POLICY_URL})",
    )
    p.add_argument(
        "--policy-backend", choices=("ws", "stub"), default="ws",
        help="ws: openpi serve_policy.py; stub: zero-action policy for plumbing tests",
    )
    p.add_argument(
        "--motion-scale", type=float, default=DEFAULT_MOTION_SCALE,
        help=(
            f"client-side scale on joint deltas "
            f"(default: {DEFAULT_MOTION_SCALE}; 1.0 = unscaled)"
        ),
    )
    p.add_argument(
        "--no-daemon-motion-scale", action="store_true",
        help=(
            "skip POSTing /v1/teleop/motion_scale (that endpoint only affects "
            "Quest input, but POSTing it surfaces the scale in the dashboard)"
        ),
    )
    p.add_argument(
        "--rate-hz", type=float, default=DEFAULT_RATE_HZ,
        help=(
            f"action commit rate (default: {DEFAULT_RATE_HZ} Hz; ~3 actions "
            "per 10-step chunk at this rate)"
        ),
    )
    p.add_argument(
        "--stream-hz", type=float, default=0.0,
        help=(
            "steady interpolated command rate to the daemon (default: 0 = off, "
            "use the proven per-action loop; set e.g. 100 to enable streaming). "
            "The policy chunk's waypoints (spaced at 1/--rate-hz) are linearly "
            "interpolated and POSTed at this fixed rate so the daemon's 250 Hz "
            "servo tracker chases a smooth, dense target instead of sparse jumpy "
            "waypoints. Set 0 to use the legacy per-action commit loop."
        ),
    )
    p.add_argument(
        "--steps-per-chunk", type=int, default=DEFAULT_STEPS_PER_CHUNK,
        help=(
            f"how many actions of each chunk to execute before re-querying "
            f"(default: {DEFAULT_STEPS_PER_CHUNK} of 10)"
        ),
    )
    p.add_argument(
        "--action-mode", choices=("delta", "absolute"), default="absolute",
        help=(
            "interpret action[:, :6] as absolute joints (matches the "
            "recorded dataset — actions[t] = state[t+1]) or deltas. "
            "Pi 0.5 / LeRobotFarmDataConfig converts to deltas internally "
            "for training but returns absolute via LiberoOutputs."
        ),
    )
    p.add_argument(
        "--no-rtc", action="store_true",
        help=(
            "disable Real-Time Chunking. By default the client sends RTC control "
            "fields so the (patched) server inpaints each new chunk to join "
            "continuously with the previous one over their overlap. --no-rtc "
            "reverts to plain independent chunks (the pre-RTC behaviour)."
        ),
    )
    p.add_argument(
        "--no-gripper", action="store_true",
        help="don't command the gripper (joints only); useful for first live test",
    )
    p.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=(
            f"hard cap on total action commits "
            f"(default: {DEFAULT_MAX_STEPS} = 60 s @ 10 Hz)"
        ),
    )
    p.add_argument(
        "--image-size", type=int, default=DEFAULT_IMAGE_SHAPE[0],
        help=(
            "square resize for both cameras before sending to the policy "
            f"(default: {DEFAULT_IMAGE_SHAPE[0]}, the paligemma vision tower size)"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "print target joints instead of POSTing to the daemon "
            "(default behavior unless --live is passed)"
        ),
    )
    p.add_argument(
        "--live", action="store_true",
        help=(
            "enable real POSTing. WITHOUT --live the script never sends a "
            "control command. ALSO needs drive_real_arm=true on the "
            "dashboard before the real arm physically follows."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tasks:
        for i, t in enumerate(load_known_tasks()):
            print(f"{i:2d}  {t}")
        return 0

    if args.dry_run and args.live:
        print("[eval] --dry-run and --live are mutually exclusive", file=sys.stderr)
        return 2
    # Default to dry-run unless --live is explicitly passed. Belt-and-braces
    # so a typo never costs us an unintended arm motion.
    dry_run = (not args.live) or args.dry_run

    parsed = urlparse(args.daemon_url)
    if parsed.scheme not in ("http", "https"):
        print(f"[eval] --daemon-url must be http(s)://; got {args.daemon_url!r}", file=sys.stderr)
        return 2

    image_shape = (int(args.image_size), int(args.image_size))
    daemon = FarmDaemonClient(args.daemon_url, image_shape=image_shape)

    # Confirm the daemon is reachable and report what it sees. This
    # alone catches the most common "I forgot to start farm serve" failure.
    try:
        snap = daemon.world()
    except Exception as exc:
        print(f"[eval] daemon at {args.daemon_url!r} unreachable: {exc}", file=sys.stderr)
        return 3
    print(
        f"[eval] daemon · backend={snap.get('backend')} · "
        f"arm_ip={snap.get('arm_ip')} · estopped={snap.get('estopped')} · "
        f"drive_real_arm={snap.get('drive_real_arm')}"
    )

    # Optional: surface the motion-scale ratio on the daemon's dashboard
    # so the operator sees what the eval client requested. NOTE this
    # endpoint scales the Quest-controller→arm mapping, not policy
    # output. We additionally scale client-side via cfg.motion_scale.
    if not args.no_daemon_motion_scale:
        try:
            daemon.set_motion_scale(args.motion_scale)
        except Exception as exc:
            print(f"[eval] WARN motion_scale POST failed (continuing): {exc}", file=sys.stderr)

    # Build the policy.
    if args.policy_backend == "stub":
        policy: Policy = StubPolicy(horizon=10, action_dim=32)
        print("[eval] policy backend = stub (zero actions)")
    else:
        try:
            policy = WebSocketPolicy(args.policy_url)
        except Exception as exc:
            print(
                f"[eval] WebSocketPolicy connection to {args.policy_url!r} failed: {exc}\n"
                "       Hint: ensure openpi serve_policy.py is running on the cluster\n"
                "             and kubectl port-forward is bridging 8000:8000 to localhost.\n"
                "       Or pass --policy-backend stub to test plumbing only.",
                file=sys.stderr,
            )
            return 4
        print(
            f"[eval] policy backend = ws ({args.policy_url}) · "
            f"server reports action_dim={policy.action_dim_reported}, "
            f"horizon={policy.action_horizon_reported}"
        )

    cfg = LoopConfig(
        task=args.task,
        rate_hz=float(args.rate_hz),
        steps_per_chunk=int(args.steps_per_chunk),
        motion_scale=float(args.motion_scale),
        max_steps=int(args.max_steps),
        action_mode=args.action_mode,
        no_gripper=bool(args.no_gripper),
        require_drive_real_arm=False,
        dry_run=dry_run,
        rtc=not bool(args.no_rtc),
        stream_hz=float(args.stream_hz),
    )
    if not dry_run:
        print(
            "[eval] LIVE mode: posting to /v1/teleop/joints. "
            f"drive_real_arm={snap.get('drive_real_arm')} — "
            "ghost arm follows always; the real arm follows only when "
            "drive_real_arm=true."
        )

    # Clean shutdown handler (SIGINT from Ctrl-C, SIGTERM from the dashboard
    # Stop button via /v1/policy/stop). We deliberately do NOT trigger an
    # e-stop here: stopping the eval should just halt the policy loop and
    # leave the arm holding its last commanded joint position. The arm is
    # position-controlled, so once we stop POSTing it simply stays put —
    # no emergency-stop state, which the operator would otherwise have to
    # clear before re-running. (Use the dashboard E-STOP button for a real
    # emergency halt.)
    def _shutdown(signum, frame):  # noqa: ARG001
        print("\n[eval] stop received — halting policy loop (no e-stop) …", file=sys.stderr)
        try:
            policy.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _policy_url = args.policy_url if args.policy_backend == "ws" else "stub"
    try:
        if cfg.stream_hz and cfg.stream_hz > 0:
            run_stream_loop(daemon=daemon, policy=policy, cfg=cfg, policy_url=_policy_url)
        else:
            run_loop(daemon=daemon, policy=policy, cfg=cfg, policy_url=_policy_url)
    finally:
        policy.close()

    # Per-chunk delta stats summary so the user can eyeball whether
    # something was clearly off (e.g. all-NaN, all-zero, huge magnitudes).
    if cfg.deltas_seen:
        all_actions = np.concatenate(cfg.deltas_seen, axis=0)
        print(
            "\n[eval] summary across "
            f"{len(cfg.deltas_seen)} chunks × {cfg.deltas_seen[0].shape[0]} steps:"
        )
        for i, name in enumerate(("j1", "j2", "j3", "j4", "j5", "j6", "grip")):
            col = all_actions[:, i]
            print(
                f"  {name:>5s}: min={col.min():+.4f} max={col.max():+.4f} "
                f"mean={col.mean():+.4f} std={col.std():.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
