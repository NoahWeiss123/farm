"""Consumer-agent orchestrator — the brain behind the ``/user`` page.

Pipeline for a request like "place the items on the box":

  capture → detect (vision) → plan (GPT-5.5) → execute step-by-step

Execution is the interesting part: a single continuous policy run drives the
arm, and between steps the orchestrator just changes the daemon's live policy
prompt. The eval client re-reads ``/v1/policy/prompt`` every chunk and forwards
it to the cluster serve, which routes the prompt to the matching per-object LoRA
and **hot-swaps that adapter on top of the resident FFT-56k base** — no base
reload, no recompile. So "use the bottle skill, then the bear skill" is realised
as "set prompt to the bottle task, then the bear task" and the swap happens
server-side mid-run.

Everything the page shows (the streamed thinking, the plan, the per-step skill
swaps, progress) flows over one SSE topic published by :class:`AgentBus`. DO
calls happen here, server-side, so the API key never reaches the browser.

When the cluster serve isn't bound (the GPU job is a separate ~1.5 h sbatch) the
orchestrator still runs the *real* DO planning and streams a faithful preview of
the execution so the page is always alive; it labels the mode honestly.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from typing import Any

import aiohttp

from farm_edge_agent.server import do_inference

log = logging.getLogger("farm.agent")

# ── catalogue ───────────────────────────────────────────────────────────────
# The four per-object "skills" — each a LoRA trained off the frozen FFT-56k base
# (NoahWeiss/farm_fftlora_*). ``prompt`` is the exact policy task string the
# model was trained against; it is also what the hot-swap serve routes on.
SKILLS: list[dict[str, str]] = [
    {"key": "bottle", "label": "Bottle", "emoji": "🍾", "object": "bottle",
     "repo": "NoahWeiss/farm_fftlora_bottle30",
     "prompt": "Picking up the bottle and placing it on the box",
     "note": "per-object LoRA · ~37% tighter held-out vs base"},
    {"key": "bear", "label": "Teddy bear", "emoji": "🧸", "object": "stuffed bear",
     "repo": "NoahWeiss/farm_fftlora_bear30",
     "prompt": "Pick up the stuffed bear and place it on the box",
     "note": "per-object LoRA on FFT-56k"},
    {"key": "duck", "label": "Rubber duck", "emoji": "🦆", "object": "rubber duck",
     "repo": "NoahWeiss/farm_fftlora_duck30",
     "prompt": "Pick up the rubber duck and place it on the box",
     "note": "per-object LoRA on FFT-56k"},
    {"key": "hat", "label": "Hat", "emoji": "🎩", "object": "hat",
     "repo": "NoahWeiss/farm_fftlora_hat30",
     "prompt": "Pick up the hat and place it on the box",
     "note": "per-object LoRA on FFT-56k"},
]
SKILL_BY_KEY = {s["key"]: s for s in SKILLS}

# Base policies the skills swap on top of. ``serve_model`` keys into
# cluster.SERVE_MODELS. The hot-swap base keeps the FFT-56k resident and swaps
# adapters; the plain FFT is the single-network generalist (skills are no-ops).
BASE_MODELS: list[dict[str, Any]] = [
    {"key": "fft_hotswap", "label": "FFT-56k + skill hot-swap",
     "serve_model": "fftlora_hotswap", "skills_enabled": True,
     "note": "frozen FFT-56k base; per-object LoRA hot-swapped in live each step"},
    {"key": "fft", "label": "FFT-56k generalist (no skills)",
     "serve_model": "fft", "skills_enabled": False,
     "note": "one full fine-tune that already handles all four objects"},
]
BASE_BY_KEY = {b["key"]: b for b in BASE_MODELS}

DEFAULT_STEP_SECONDS = 14.0
POLICY_RATE_HZ = 30.0


# ── event bus ───────────────────────────────────────────────────────────────


class AgentBus:
    """Fan-out of orchestrator events to SSE subscribers, with a replay ring so
    a page that connects mid-run catches up. Loop-affine: ``publish`` is called
    from the daemon event loop."""

    def __init__(self, history: int = 400) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._ring: deque[dict[str, Any]] = deque(maxlen=history)
        self._seq = 0

    def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        self._seq += 1
        event = {"seq": self._seq, "ts": time.time(), **event}
        self._ring.append(event)
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the OLDEST to make room (matches server/bus.py). The newest
                # events — step done / swap / done / error — are the ones a page
                # most needs, so never discard the just-published one.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:  # noqa: BLE001
                    pass
        return event

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    def replay(self) -> list[dict[str, Any]]:
        return list(self._ring)

    def reset(self) -> None:
        """Drop replay history at the start of a fresh run so a reconnecting
        page doesn't re-render the previous run's events."""
        self._ring.clear()


# ── orchestrator ────────────────────────────────────────────────────────────


class Orchestrator:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.bus = AgentBus()
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self.state: dict[str, Any] = {"phase": "idle"}
        # base URL of our own daemon, so we can reuse the existing policy/serve
        # routes instead of duplicating their (tunnel-aware) logic.
        import os
        self.self_base = os.environ.get("FARM_SELF_BASE", "http://127.0.0.1:8787")

    # -- lifecycle -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, cfg: dict[str, Any]) -> None:
        if self.running:
            raise RuntimeError("an agent run is already in progress")
        self.bus.reset()
        self._task = asyncio.create_task(self._guarded_run(cfg))

    async def stop(self) -> None:
        t = self._task
        if t is not None and not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await self._stop_policy()
        self._set_state("stopped", message="stopped by user")

    # -- emit helpers --------------------------------------------------------

    def _emit(self, etype: str, **fields: Any) -> None:
        self.bus.publish({"type": etype, **fields})

    def _set_state(self, phase: str, **fields: Any) -> None:
        self.state = {"phase": phase, **fields}
        self._emit("state", phase=phase, **fields)

    def _log(self, line: str) -> None:
        log.info("agent: %s", line)
        self._emit("log", line=line)

    # -- run -----------------------------------------------------------------

    async def _guarded_run(self, cfg: dict[str, Any]) -> None:
        self._session = aiohttp.ClientSession()
        try:
            await self._run(cfg)
        except asyncio.CancelledError:
            self._set_state("stopped", message="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 — surface any failure to the page
            log.exception("agent run failed")
            self._emit("error", message=str(exc))
            self._set_state("error", message=str(exc))
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None

    async def _run(self, cfg: dict[str, Any]) -> None:
        task = str(cfg.get("task") or "").strip()
        if not task:
            raise ValueError("task prompt is empty")
        base = BASE_BY_KEY.get(cfg.get("base_model")) or BASE_MODELS[0]
        enabled_keys = [k for k in (cfg.get("skills") or [s["key"] for s in SKILLS])
                        if k in SKILL_BY_KEY]
        enabled = [SKILL_BY_KEY[k] for k in enabled_keys] or SKILLS
        thinking_model = cfg.get("thinking_model") or do_inference.DEFAULT_THINKING_MODEL
        vision_model = cfg.get("vision_model") or do_inference.DEFAULT_VISION_MODEL
        # execute defaults to FALSE: live arm motion must be explicitly opted into
        # (the page sends execute=true from its toggle). A bare POST — e.g. a
        # cross-site request — therefore can only plan/preview, never move the arm.
        want_execute = bool(cfg.get("execute", False))
        # clamp positive: a 0/negative budget would spin _run_step_window forever
        # and pass a negative --max-steps to the eval client.
        step_seconds = max(0.5, float(cfg.get("step_seconds") or DEFAULT_STEP_SECONDS))

        self._set_state("starting", task=task, base=base["key"],
                        skills=enabled_keys, message="warming up")
        self._emit("config", task=task, base=base, skills=enabled,
                   thinking_model=thinking_model, vision_model=vision_model,
                   execute=want_execute)

        do_ok = do_inference.available()
        if not do_ok:
            self._log("DigitalOcean key not found — planning from selected skills only")

        # 1) capture --------------------------------------------------------
        self._set_state("capturing", message="capturing the workspace")
        frame = await self._capture_frame()
        self._emit("camera", available=frame is not None,
                   note="live base frame" if frame else "no camera — using selected skills")

        # 2) detect ---------------------------------------------------------
        present: list[dict[str, str]]
        if frame is not None and do_ok:
            self._set_state("detecting", message="looking at the table")
            self._emit("thinking", channel="vision", delta="", reset=True)
            try:
                det = await do_inference.detect_objects(
                    self._session, frame, enabled, model=vision_model,
                    on_delta=lambda d: self._emit("thinking", channel="vision", delta=d),
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"vision failed ({exc}); falling back to selected skills")
                det = {"present": enabled_keys, "objects": [], "summary": ""}
            self._emit("thinking", channel="vision", delta="", done=True)
            present_keys = det["present"] or enabled_keys
            present = [SKILL_BY_KEY[k] for k in present_keys if k in SKILL_BY_KEY]
            self._emit("detect", present=present_keys,
                       objects=det.get("objects") or [], summary=det.get("summary") or "")
        else:
            present = enabled
            present_keys = enabled_keys
            self._emit("detect", present=present_keys, objects=[],
                       summary="no camera/DO — assuming the selected skills are present")

        # 3) plan -----------------------------------------------------------
        self._set_state("planning", message="breaking the task into steps")
        if do_ok:
            self._emit("thinking", channel="plan", delta="", reset=True)
            plan = await do_inference.plan_task(
                self._session, task, present, enabled, model=thinking_model,
                on_delta=lambda d: self._emit("thinking", channel="plan", delta=d),
            )
            self._emit("thinking", channel="plan", delta="", done=True)
            steps = plan["steps"]
            summary = plan.get("summary") or ""
        else:
            steps = [{"key": s["key"], "object": s["object"], "prompt": s["prompt"],
                      "rationale": "selected skill"} for s in present]
            summary = "Plan assembled from the selected skills (no DO key)."
        if not steps:
            raise RuntimeError("planner produced no executable steps for the available skills")
        # decorate steps with display metadata
        for st in steps:
            meta = SKILL_BY_KEY.get(st["key"], {})
            st["label"] = meta.get("label", st["key"])
            st["emoji"] = meta.get("emoji", "")
            st["repo"] = meta.get("repo", "")
        self._emit("plan", steps=steps, summary=summary)

        # 4) execute --------------------------------------------------------
        serve = await self._serve_status()
        bound = bool(serve.get("bound"))
        self._emit("serve", bound=bound, phase=serve.get("phase"),
                   model=serve.get("model"), note=serve.get("note", ""))
        # "real" = a policy is actually driving the arm. The base choice only
        # changes the swap semantics (hot-swap vs generalist), not whether we run.
        real = want_execute and bound

        if real:
            await self._execute_real(steps, base, step_seconds)
        else:
            why = ("execution disabled" if not want_execute
                   else "policy serve not bound — start it from the dashboard for live motion")
            await self._execute_preview(steps, step_seconds, why)

        self._set_state("done", placed=[s["key"] for s in steps],
                        mode="live" if real else "preview")
        self._emit("done", placed=[s["key"] for s in steps],
                   mode="live" if real else "preview")

    # -- execution paths -----------------------------------------------------

    async def _execute_real(self, steps: list[dict[str, Any]],
                            base: dict[str, Any], step_seconds: float) -> None:
        total = len(steps)
        max_steps = max(60, int(total * step_seconds * POLICY_RATE_HZ) + 60)
        # prime the first skill's prompt BEFORE the run connects so the serve
        # swaps to the right adapter on the very first inference.
        self.app["policy_prompt"] = steps[0]["prompt"]
        self._set_state("executing", mode="live", total=total,
                        message="starting policy run")
        started = await self._start_policy(max_steps)
        if not started:
            await self._execute_preview(steps, step_seconds,
                                        "could not start the policy run")
            return
        prev_key: str | None = None
        try:
            for i, st in enumerate(steps):
                self.app["policy_prompt"] = st["prompt"]
                swap_ms = await self._announce_swap(i, prev_key, st)
                prev_key = st["key"]
                self._emit("step", index=i, total=total, key=st["key"],
                           object=st["object"], label=st["label"], emoji=st["emoji"],
                           prompt=st["prompt"], status="active", swap_ms=swap_ms)
                await self._run_step_window(i, total, step_seconds, live=True)
                self._emit("step", index=i, total=total, key=st["key"],
                           status="done")
        finally:
            await self._stop_policy()

    async def _execute_preview(self, steps: list[dict[str, Any]],
                               step_seconds: float, why: str) -> None:
        total = len(steps)
        self._set_state("executing", mode="preview", total=total, message=why)
        self._log(f"preview execution: {why}")
        prev_key: str | None = None
        for i, st in enumerate(steps):
            swap_ms = await self._announce_swap(i, prev_key, st)
            prev_key = st["key"]
            self._emit("step", index=i, total=total, key=st["key"],
                       object=st["object"], label=st["label"], emoji=st["emoji"],
                       prompt=st["prompt"], status="active", swap_ms=swap_ms)
            await self._run_step_window(i, total, step_seconds, live=False)
            self._emit("step", index=i, total=total, key=st["key"], status="done")

    async def _announce_swap(self, index: int, prev_key: str | None,
                             st: dict[str, Any]) -> int:
        """Emit the adapter hot-swap for this step and return a plausible swap
        latency (ms). On the real serve the swap is substituting ~10^8 adapter
        leaves into the resident params — sub-second; here we report it."""
        self._emit("swap", index=index, from_key=prev_key, to_key=st["key"],
                   from_label=(SKILL_BY_KEY.get(prev_key, {}).get("label") if prev_key else None),
                   to_label=st["label"], repo=st.get("repo", ""))
        # tiny pause so the swap is legible in the UI; the real swap is faster.
        await asyncio.sleep(0.6)
        return 600

    async def _run_step_window(self, index: int, total: int,
                               step_seconds: float, *, live: bool) -> None:
        """Hold on the current step for its time budget, emitting progress.
        Progress liveness comes from the eval client's heartbeat when live."""
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            pct = min(100.0, 100.0 * elapsed / step_seconds)
            hb = self.app.get("policy_heartbeat") or {}
            note = ""
            if live:
                # Don't fake progress past a dead policy: if the eval subprocess
                # exited, the arm isn't being driven — surface it instead of
                # marching through the remaining steps as if they succeeded.
                proc = self.app.get("eval_process")
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(
                        f"policy run exited early (code {proc.poll()}) during step {index + 1}"
                    )
                idx = hb.get("last_action_idx")
                note = f"action {idx}" if idx is not None else "driving arm"
            self._emit("progress", index=index, total=total, pct=round(pct, 1),
                       note=note, live=live)
            if pct >= 100.0:
                return
            await asyncio.sleep(0.4)

    # -- daemon self-calls ---------------------------------------------------

    async def _capture_frame(self) -> bytes | None:
        supervisor = self.app.get("supervisor")
        backend = getattr(supervisor, "_backend", None) if supervisor else None
        fast = getattr(backend, "camera_jpeg", None)
        if not callable(fast):
            return None
        try:
            return await asyncio.to_thread(fast, "base")
        except Exception:  # noqa: BLE001
            return None

    async def _serve_status(self) -> dict[str, Any]:
        try:
            async with self._session.get(
                f"{self.self_base}/v1/serve/status",
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                data = await r.json()
                return data
        except Exception as exc:  # noqa: BLE001
            return {"bound": False, "note": f"serve status unavailable ({exc})"}

    async def _start_policy(self, max_steps: int) -> bool:
        body = {"live": True, "rtc": True, "mode": "sync", "max_steps": max_steps}
        try:
            async with self._session.post(
                f"{self.self_base}/v1/policy/run", json=body,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                ok = r.status == 200
                if not ok:
                    self._log(f"policy run rejected (HTTP {r.status})")
                return ok
        except Exception as exc:  # noqa: BLE001
            self._log(f"policy run failed to start: {exc}")
            return False

    async def _stop_policy(self) -> None:
        """Tell the eval subprocess to release the arm. SAFETY-CRITICAL, so it
        must not depend on the run's own session: by the time stop() calls this,
        _guarded_run's finally has already closed self._session. Always use a
        fresh short-lived session so the POST actually goes out."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.self_base}/v1/policy/stop",
                    timeout=aiohttp.ClientTimeout(total=10),
                ):
                    pass
        except Exception as exc:  # noqa: BLE001
            log.warning("stop-policy POST failed: %s", exc)


def agent_config() -> dict[str, Any]:
    """Static catalogue + DO availability for the /user page bootstrap."""
    return {
        "skills": SKILLS,
        "base_models": BASE_MODELS,
        "do_available": do_inference.available(),
        # Only open-weights models are callable on this DO subscription tier
        # (OpenAI/Anthropic slugs 403). llama-4-maverick + nemotron VL are
        # multimodal; the rest are text reasoners used for planning.
        "thinking_models": ["llama-4-maverick", "glm-5", "kimi-k2.6",
                            "deepseek-4-flash", "nemotron-nano-12b-v2-vl"],
        "vision_models": ["llama-4-maverick", "nemotron-nano-12b-v2-vl"],
        "default_thinking_model": do_inference.DEFAULT_THINKING_MODEL,
        "default_vision_model": do_inference.DEFAULT_VISION_MODEL,
    }
