"""Reference eval client for the FARM UF850 π0.5 policy — built on the REAL
openpi-client so the inference path is byte-identical to openpi's own examples,
now with **Real-Time Chunking (RTC)** so consecutive action chunks join smoothly
instead of overshooting and snapping back at every chunk boundary.

Why this exists (vs the homegrown ``eval_pi05.py``): to be *certain* the
plumbing matches what ``serve_policy.py`` expects, this uses openpi's actual
code for everything that talks to the model:

  * ``openpi_client.websocket_client_policy.WebsocketClientPolicy`` — the wire
    protocol (msgpack-numpy, connect handshake), not a reimplementation.
  * ``openpi_client.image_tools.resize_with_pad`` / ``convert_to_uint8`` — the
    exact training-time image preprocessing.

Real-Time Chunking (Black et al., arXiv:2506.07339; the patched server runs PI's
guidance). A chunked flow policy predicts H absolute joint waypoints per
inference. Naively re-planning every chunk makes the arm lurch to the new
chunk's start and snap back — the boundary discontinuity the operator sees as
"overshoot then violently back." RTC fixes it: the server keeps the previous raw
chunk and guides the next one's flow-matching denoiser to stay continuous over
their overlap (freezing the actions that execute during the inference delay,
softly inpainting the rest). The client just has to (a) re-plan with OVERLAP —
re-infer before the current chunk is exhausted — and (b) tell the server how the
two chunks line up via three obs-dict fields the patched ``Policy.infer`` pops:

    rtc_reset  — drop server state (first chunk of a run)
    rtc_offset — action-steps elapsed between the previous chunk's obs and this
                 one (= how far to shift the stored chunk to align them)
    rtc_delay  — inference delay in steps (the hard-frozen prefix length)

Execution is double-buffered + timed-waypoint: one chunk plays out at the
control rate while the next infers in a background thread; action[i] is committed
at wall time ``obs_time + (i+1)·period`` and stale actions (whose time already
passed during inference) are skipped. This hides the ~80–120 ms inference behind
motion, so there are no pauses AND — with RTC — no rubber-banding.

Observation format is openpi's LIBERO layout (our ``LeRobotFarmDataConfig``
reuses ``LiberoInputs``):

    {"observation/image":      uint8 (224,224,3),
     "observation/wrist_image":uint8 (224,224,3),
     "observation/state":      float32 (7,)  # 6 joints rad + gripper 0..1,
     "prompt":                 "<exact trained task string>"}

KEY vs eval_pi05.py: the prompt is the EXACT ``--task`` you pass — we never read
the daemon's dashboard prompt, and we never scale/clip the action (absolute
joint targets go to the arm verbatim; the daemon upsamples them to a smooth
250 Hz stream). π0.5's tokenizer is case-sensitive, so use a task string verbatim
from the trained set (``--list-tasks``).

Run (dry-run is the default; pass --live to move the arm):

    python model/eval_pi05_ref.py --task "Pick up the stuffed bear and place it on the box"
    python model/eval_pi05_ref.py --task "Pick up the stuffed bear and place it on the box" --live
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time
from typing import Any

import numpy as np
import requests
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from PIL import Image

# The EXACT strings the multiobject GSE-robust model trained on (from
# datasets/lerobot/farm_uf850_multiobject/meta/tasks.jsonl). π0.5 is
# language-conditioned and case-sensitive — use one of these verbatim.
TRAINED_TASKS = [
    "Picking up the bottle and placing it on the box",
    "Pick up the stuffed bear and place it on the box",
    "Pick up the rubber duck and place it on the box",
    "Pick up the hat and place it on the box",
]

IMG = 224  # PaliGemma vision-tower input


def fetch_observation(daemon: str, prompt: str, sess: requests.Session) -> tuple[dict, dict]:
    """Build the openpi obs dict from the daemon. Returns (obs, world_snapshot)."""
    world = sess.get(f"{daemon}/v1/world", timeout=3).json()
    joints = world.get("joints") or []
    if len(joints) < 6:
        raise RuntimeError(f"daemon snapshot has {len(joints)} joints, need 6")
    grip = world.get("gripper_pos")
    grip = float(grip) if isinstance(grip, (int, float)) else 0.0
    state = np.asarray([float(joints[i]) for i in range(6)] + [grip], dtype=np.float32)

    def cam(name: str) -> np.ndarray:
        r = sess.get(f"{daemon}/v1/cameras/{name}.jpg", timeout=3)
        r.raise_for_status()
        rgb = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.uint8)
        # openpi's exact preprocessing: aspect-preserving resize + pad to 224².
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, IMG, IMG))

    obs = {
        "observation/image": cam("base"),
        "observation/wrist_image": cam("wrist"),
        "observation/state": state,
        "prompt": prompt,
    }
    return obs, world


def post_joints(daemon: str, joints: np.ndarray, gripper: float, sess: requests.Session) -> dict:
    body = {"joints": [float(j) for j in joints], "gripper": float(gripper)}
    r = sess.post(f"{daemon}/v1/teleop/joints", json=body, timeout=2.0)
    if r.status_code >= 400:
        raise RuntimeError(f"POST /v1/teleop/joints → {r.status_code}: {r.text}")
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="", help="EXACT trained task string (see --list-tasks)")
    ap.add_argument("--list-tasks", action="store_true", help="print the trained tasks and exit")
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=8000)
    ap.add_argument("--daemon-url", default="http://127.0.0.1:8787")
    ap.add_argument("--action-horizon", type=int, default=10,
                    help="initial chunk-length guess (the real H is read from the first chunk)")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--rate-hz", type=float, default=30.0,
                    help="control rate — MUST match the training timestep (30 Hz); the daemon "
                         "upsamples to a smooth 250 Hz, so do NOT raise this to 'go faster'.")
    ap.add_argument("--rtc", dest="rtc", action="store_true", default=True,
                    help="Real-Time Chunking seam smoothing (default ON — the fix for "
                         "chunk-boundary overshoot)")
    ap.add_argument("--no-rtc", dest="rtc", action="store_false",
                    help="disable RTC (vanilla chunking — will overshoot at seams)")
    ap.add_argument("--live", action="store_true", help="POST joint targets to the arm (default: dry-run)")
    args = ap.parse_args()

    if args.list_tasks:
        for t in TRAINED_TASKS:
            print(t)
        return 0
    if not args.task:
        print("--task is required (use --list-tasks to see the exact trained strings)", file=sys.stderr)
        return 2
    if args.task not in TRAINED_TASKS:
        print(f"[ref] WARNING: --task {args.task!r} is NOT an exact trained string.\n"
              f"       π0.5's tokenizer is case-sensitive — a near-miss degrades conditioning.\n"
              f"       Use one of (verbatim):", file=sys.stderr)
        for t in TRAINED_TASKS:
            print(f"         {t!r}", file=sys.stderr)

    task = args.task
    rate_hz = max(1.0, args.rate_hz)
    period = 1.0 / rate_hz
    rtc_on = bool(args.rtc)

    main_sess = requests.Session()   # main thread: estop checks + POST joints
    bg_sess = requests.Session()     # bg thread: obs fetch (separate session = thread-safe)

    # 1) daemon reachable + report what it sees
    try:
        w = main_sess.get(f"{args.daemon_url}/v1/world", timeout=3).json()
    except Exception as exc:
        print(f"[ref] daemon at {args.daemon_url} unreachable: {exc}", file=sys.stderr)
        return 3
    print(f"[ref] daemon: backend={w.get('backend')} arm_ip={w.get('arm_ip')} "
          f"estopped={w.get('estopped')} drive_real_arm={w.get('drive_real_arm')}")
    if args.live and not w.get("drive_real_arm"):
        print("[ref] NOTE: drive_real_arm is False — the ghost arm will track but the REAL arm won't move "
              "until you flip 'drive real arm' on in the dashboard.", file=sys.stderr)

    # 2) connect to the policy server with the REAL openpi client
    print(f"[ref] connecting to policy ws://{args.policy_host}:{args.policy_port} …")
    base = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    print(f"[ref] connected · server metadata = {base.get_server_metadata()}")
    print(f"[ref] prompt (EXACT, no dashboard override): {task!r}")
    print(f"[ref] mode: {'LIVE — arm WILL move' if args.live else 'DRY-RUN — prints targets, no motion'} · "
          f"rtc={'ON' if rtc_on else 'off'} · rate={rate_hz:.0f}Hz (daemon upsamples to 250Hz)")

    # ── Real-Time Chunking state ───────────────────────────────────────────────
    # Read by the bg inference thread when it is spawned (mid-chunk); written by
    # the main thread only AFTER joining that thread (end-of-chunk) — so there is
    # no concurrent access. `prev_obs_time` anchors the chunk currently executing;
    # `delay_steps` is the rolling inference latency in action-steps (frozen
    # prefix); `horizon` is the chunk length H.
    rtc_state: dict[str, Any] = {
        "prev_obs_time": None,
        "delay_steps": max(1, int(round(0.12 * rate_hz))),
        "horizon": int(args.action_horizon),
    }

    def build_rtc(obs_wall_t: float, *, reset: bool) -> dict[str, Any]:
        """RTC control fields for one inference (empty dict ⇒ vanilla chunking).
        Ported verbatim from the validated eval_pi05.py; matches what the patched
        server's Policy.infer pops off the obs dict."""
        if not rtc_on:
            return {}
        if reset or rtc_state["prev_obs_time"] is None:
            return {"rtc_reset": True}
        h = int(rtc_state["horizon"])
        offset = int(round((obs_wall_t - rtc_state["prev_obs_time"]) * rate_hz))
        offset = max(1, min(h - 1, offset))          # clamp ⇒ chunks always overlap
        overlap = h - offset
        delay = max(0, min(overlap, int(rtc_state["delay_steps"])))
        return {"rtc_offset": offset, "rtc_delay": delay}

    def fetch_and_infer(*, reset: bool = False) -> dict[str, Any]:
        """One (obs fetch → RTC-guided infer) pass. Records obs_time so each
        predicted action can be committed at its intended wall-clock time."""
        out: dict[str, Any] = {}
        obs_wall_t = time.perf_counter()
        try:
            obs, world = fetch_observation(args.daemon_url, task, bg_sess)
        except Exception as exc:
            out["error"] = f"obs fetch failed: {exc}"
            return out
        if world.get("estopped"):
            out["estopped"] = True
            return out
        rtc = build_rtc(obs_wall_t, reset=reset)
        try:
            t0 = time.perf_counter()
            result = base.infer({**obs, **rtc})
            timing = result.get("policy_timing", {}) if isinstance(result, dict) else {}
            out["infer_ms"] = float(timing.get("infer_ms", (time.perf_counter() - t0) * 1000))
            out["actions"] = np.asarray(result["actions"], dtype=np.float32)
            out["obs_time"] = obs_wall_t
            out["rtc"] = rtc
            out["ok"] = True
        except Exception as exc:
            out["ok"] = False
            out["error"] = f"infer failed: {exc}"
        return out

    # 3) bootstrap — need the first chunk before any motion.  reset=True drops any
    # RTC state a previous run left on the server.
    current = fetch_and_infer(reset=True)
    if current.get("estopped"):
        print("[ref] daemon is e-stopped at start — aborting.", file=sys.stderr)
        return 4
    if not current.get("ok"):
        print(f"[ref] first inference FAILED: {current.get('error')}", file=sys.stderr)
        return 5

    action_chunk = current["actions"]
    if action_chunk.ndim != 2 or action_chunk.shape[0] < 1:
        print(f"[ref] unexpected action chunk shape {action_chunk.shape}", file=sys.stderr)
        return 6
    # Treat the bootstrap obs as "freshly captured now" so the first chunk uses
    # all H predictions instead of skipping the ones that elapsed during the
    # cold-start JAX compile (which can be seconds).
    obs_time = time.perf_counter()
    H = int(action_chunk.shape[0])
    rtc_state["horizon"] = H
    rtc_state["prev_obs_time"] = obs_time
    rtc_state["delay_steps"] = max(1, int(round(current["infer_ms"] / 1000 * rate_hz)))
    print(f"[ref] first inference {current['infer_ms']:.0f} ms (incl. one-time JIT compile) · "
          f"chunk H={H} · steady-state ~80–120 ms.")

    log_every = max(1, H)

    # ── Double-buffered pipelined timed-waypoint loop ─────────────────────────
    step = 0
    bg_thread: threading.Thread | None = None
    holder: dict[str, dict[str, Any]] = {"r": {}}

    def spawn_bg() -> None:
        nonlocal bg_thread
        holder["r"] = {}
        bg_thread = threading.Thread(
            target=lambda: holder["r"].update(fetch_and_infer()),
            daemon=True, name="bg-infer",
        )
        bg_thread.start()

    aborted = False
    while step < args.max_steps and not aborted:
        # absolute joint targets, verbatim (no scaling): action[:6] rad, action[6] gripper 0..1
        plan = [
            (action_chunk[i, :6], float(np.clip(action_chunk[i, 6], 0.0, 1.0)) if action_chunk.shape[1] > 6 else 0.0)
            for i in range(H)
        ]
        # Skip actions whose intended wall time already passed (during inference).
        elapsed = time.perf_counter() - obs_time
        skip = max(0, min(H - 1, int(round(elapsed * rate_hz))))
        remaining = H - skip
        spawn_after = skip + max(0, remaining // 2)   # re-plan ~halfway ⇒ overlap for RTC

        for i in range(skip, H):
            if step >= args.max_steps:
                break
            if bg_thread is None and i >= spawn_after:
                spawn_bg()
            # Timed-waypoint commit: sleep until this action's absolute wall time.
            wait_for = (obs_time + (i + 1) * period) - time.perf_counter()
            if wait_for > 0:
                time.sleep(wait_for)

            joints, gripper = plan[i]
            if args.live:
                try:
                    resp = post_joints(args.daemon_url, joints, gripper, main_sess)
                except Exception as exc:
                    msg = str(exc)
                    if "409" in msg or "estop" in msg.lower():
                        print(f"[ref] POST refused (likely e-stop): {msg}", file=sys.stderr)
                    else:
                        print(f"[ref] POST failed: {msg}", file=sys.stderr)
                    aborted = True
                    break
                if isinstance(resp, dict) and resp.get("drive_real_arm") is False and w.get("drive_real_arm"):
                    print("[ref] drive_real_arm turned OFF mid-run; ghost tracks, real arm stopped.",
                          file=sys.stderr)
            if step % log_every == 0:
                tag = "LIVE " if args.live else "DRY  "
                r = current.get("rtc") or {}
                rtc_tag = "reset" if r.get("rtc_reset") else (f"off={r.get('rtc_offset')},dly={r.get('rtc_delay')}"
                                                              if r else "—")
                print(f"  {tag}step {step:4d} (chunk idx {i:2d}) "
                      f"joints=[{', '.join(f'{j:+.3f}' for j in joints)}]  grip={gripper:.2f}  rtc[{rtc_tag}]")
            step += 1

        if aborted or step >= args.max_steps:
            break

        # Pick up the pipelined next chunk (or infer synchronously if bg never spawned).
        if bg_thread is not None:
            bg_thread.join()
            nxt = holder["r"]
            bg_thread = None
        else:
            nxt = fetch_and_infer()

        if nxt.get("estopped"):
            print("[ref] daemon is e-stopped — aborting loop.", file=sys.stderr)
            break
        if not nxt.get("ok"):
            print(f"[ref] infer failed: {nxt.get('error')} — retrying once…", file=sys.stderr)
            time.sleep(0.5)
            nxt = fetch_and_infer()
            if not nxt.get("ok"):
                print("[ref] retry also failed; aborting.", file=sys.stderr)
                break

        current = nxt
        action_chunk = nxt["actions"]
        obs_time = nxt["obs_time"]
        H = int(action_chunk.shape[0])
        # This chunk is now executing ⇒ it anchors the NEXT background inference.
        rtc_state["prev_obs_time"] = obs_time
        rtc_state["horizon"] = H
        rtc_state["delay_steps"] = max(1, int(round(nxt["infer_ms"] / 1000 * rate_hz)))
        log_every = max(1, H)

    print(f"[ref] done · {step} steps committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
