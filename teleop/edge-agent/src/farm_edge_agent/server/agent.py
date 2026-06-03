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
import base64
import contextlib
import logging
import time
from collections import deque
from typing import Any

import aiohttp

from farm_edge_agent.server import abilities, do_inference, samples

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
DEFAULT_CONFIRM_THRESHOLD = 0.6


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
        base = BASE_BY_KEY.get(cfg.get("base_model")) or BASE_MODELS[0]
        # One multimodal model does both the planning and the image recognition.
        self._model = (cfg.get("model") or cfg.get("thinking_model")
                       or do_inference.DEFAULT_THINKING_MODEL)
        self._vision_model = self._model
        self._confirm_threshold = max(
            0.0, min(1.0, float(cfg.get("confirm_threshold") or DEFAULT_CONFIRM_THRESHOLD)))
        # Images on = perceive + confirm from the camera (or sample footage when
        # there's no live camera). Off = no-image mode: plan from the task text.
        self._use_images = bool(cfg.get("use_images", True))
        self._sample_id = cfg.get("sample_episode") or None
        self._total = 1
        want_execute = bool(cfg.get("execute", False))
        step_seconds = max(0.5, float(cfg.get("step_seconds") or DEFAULT_STEP_SECONDS))

        # Ability re-run: skip vision + planning, replay the saved steps.
        ability_id = cfg.get("ability_id")
        if ability_id:
            ab = abilities.get_ability(str(ability_id))
            if ab is None:
                raise RuntimeError(f"ability {ability_id!r} not found")
            steps = self._decorate(ab.get("steps") or [])
            if not steps:
                raise RuntimeError("this ability has no steps")
            task = ab.get("task") or ab.get("name") or "ability"
            base = BASE_BY_KEY.get(ab.get("base_model")) or base
            self._set_state("starting", task=task, base=base["key"], ability=ab.get("name"),
                            message=f"running ability: {ab.get('name')}")
            self._emit("config", task=task, base=base, skills=steps,
                       model=self._model, execute=want_execute,
                       ability={"id": ab.get("id"), "name": ab.get("name")})
            self._emit_workflow(task, steps, ability=True)
            self._node("trigger", "done")
            self._emit("plan", steps=steps, summary=ab.get("summary") or "", ability=ab.get("name"))
            await self._dispatch(steps, base, step_seconds, want_execute=want_execute)
            return

        # Fresh plan: perceive, plan, then execute.
        task = str(cfg.get("task") or "").strip()
        if not task:
            raise ValueError("task prompt is empty")
        enabled_keys = [k for k in (cfg.get("skills") or [s["key"] for s in SKILLS])
                        if k in SKILL_BY_KEY]
        enabled = [SKILL_BY_KEY[k] for k in enabled_keys] or SKILLS

        self._set_state("starting", task=task, base=base["key"],
                        skills=enabled_keys, message="warming up")
        self._emit("config", task=task, base=base, skills=enabled,
                   model=self._model, execute=want_execute)
        # The graph skeleton shows immediately, even with no arm or camera.
        self._emit_workflow(task, None, ability=False)
        self._node("trigger", "done")

        do_ok = do_inference.available()
        if not do_ok:
            self._log("no inference key found, planning from the selected skills only")

        # perceive
        self._node("perceive", "active")
        self._set_state("capturing", message="looking at the workspace")
        frame = await self._observe(0.04, "Looking at the table") if self._use_images else None
        self._emit("camera", available=frame is not None,
                   note=("live frame" if frame else
                         ("no-image mode, planning from your description" if not self._use_images
                          else "no camera or sample, using selected skills")))
        if frame is not None and do_ok:
            self._set_state("detecting", message="reading the scene")
            self._emit("thinking", channel="vision", delta="", reset=True)
            try:
                det = await do_inference.detect_objects(
                    self._session, frame, enabled, model=self._model,
                    on_delta=lambda d: self._emit("thinking", channel="vision", delta=d),
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"vision failed ({exc}); falling back to selected skills")
                det = {"present": enabled_keys, "objects": [], "summary": ""}
            self._emit("thinking", channel="vision", delta="", done=True)
            present_keys = det["present"] or enabled_keys
            self._emit("detect", present=present_keys,
                       objects=det.get("objects") or [], summary=det.get("summary") or "")
        else:
            present_keys = enabled_keys
            self._emit("detect", present=present_keys, objects=[],
                       summary=("No-image mode: taking your task at face value."
                                if not self._use_images
                                else "No camera or sample selected, using the selected skills."))
        present = [SKILL_BY_KEY[k] for k in present_keys if k in SKILL_BY_KEY] or enabled
        self._node("perceive", "done")

        # plan
        self._node("plan", "active")
        self._set_state("planning", message="breaking the task into steps")
        if do_ok:
            self._emit("thinking", channel="plan", delta="", reset=True)
            plan = await do_inference.plan_task(
                self._session, task, present, enabled, model=self._model,
                on_delta=lambda d: self._emit("thinking", channel="plan", delta=d),
            )
            self._emit("thinking", channel="plan", delta="", done=True)
            steps = self._decorate(plan["steps"])
            summary = plan.get("summary") or ""
        else:
            steps = self._decorate([{"key": s["key"], "rationale": "selected skill"} for s in present])
            summary = "Plan assembled from the selected skills."
        if not steps:
            raise RuntimeError("the planner produced no runnable steps for the available skills")
        self._node("plan", "done")
        # Now emit the FULL graph (step + confirm nodes); the page rebuilds the
        # graph on a workflow event, so re-assert the upstream nodes as done.
        self._emit_workflow(task, steps, ability=False)
        self._node("trigger", "done")
        self._node("perceive", "done")
        self._node("plan", "done")
        self._emit("plan", steps=steps, summary=summary)

        await self._dispatch(steps, base, step_seconds, want_execute=want_execute)

    # -- workflow graph ------------------------------------------------------

    def _decorate(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fill each step with display + policy metadata from the skill catalogue."""
        out: list[dict[str, Any]] = []
        for st in steps or []:
            if not isinstance(st, dict) or "key" not in st:
                continue
            meta = SKILL_BY_KEY.get(st["key"], {})
            out.append({
                **st,
                "object": st.get("object") or meta.get("object", st["key"]),
                "label": st.get("label") or meta.get("label", st["key"]),
                "emoji": st.get("emoji") or meta.get("emoji", ""),
                "prompt": st.get("prompt") or meta.get("prompt", ""),
                "repo": st.get("repo") or meta.get("repo", ""),
                "rationale": st.get("rationale", ""),
            })
        return out

    def _emit_workflow(self, task: str, steps: list[dict[str, Any]] | None, *, ability: bool) -> None:
        """Publish the node-graph spec the page draws: trigger, then (for a fresh
        plan) perceive and plan, then a skill+confirm pair per object, then done."""
        nodes: list[dict[str, Any]] = [
            {"id": "trigger", "kind": "trigger", "title": "Task", "detail": self._short(task)},
        ]
        if not ability:
            nodes.append({"id": "perceive", "kind": "perceive", "title": "Look", "detail": "scan the table"})
            nodes.append({"id": "plan", "kind": "plan", "title": "Plan", "detail": "break into steps"})
        for i, s in enumerate(steps or []):
            nodes.append({"id": f"step-{i}", "kind": "skill",
                          "title": s.get("label") or s.get("object") or s.get("key"),
                          "emoji": s.get("emoji", ""), "skill": s.get("key"),
                          "detail": s.get("repo", "")})
            nodes.append({"id": f"confirm-{i}", "kind": "confirm", "title": "Confirm",
                          "detail": f"is the {s.get('object') or 'object'} placed?"})
        nodes.append({"id": "done", "kind": "done", "title": "Done", "detail": ""})
        edges = [[nodes[k]["id"], nodes[k + 1]["id"]] for k in range(len(nodes) - 1)]
        self._emit("workflow", nodes=nodes, edges=edges, ability=bool(ability))

    def _node(self, nid: str, status: str, **extra: Any) -> None:
        self._emit("node", id=nid, status=status, **extra)

    @staticmethod
    def _short(s: str | None, n: int = 64) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 1] + "..."

    # -- execution -----------------------------------------------------------

    async def _dispatch(self, steps: list[dict[str, Any]], base: dict[str, Any],
                        step_seconds: float, *, want_execute: bool) -> None:
        serve = await self._serve_status()
        bound = bool(serve.get("bound"))
        self._emit("serve", bound=bound, phase=serve.get("phase"),
                   model=serve.get("model"), note=serve.get("note", ""))
        live = want_execute and bound
        if not live:
            self._log("execution off, plan only" if not want_execute
                      else "policy serve not running, previewing the plan")
        await self._execute(steps, step_seconds, live=live)
        mode = "live" if live else "preview"
        self._node("done", "done")
        self._set_state("done", placed=[s["key"] for s in steps], mode=mode)
        self._emit("done", placed=[s["key"] for s in steps], mode=mode)

    async def _execute(self, steps: list[dict[str, Any]], step_seconds: float, *, live: bool) -> None:
        total = len(steps)
        self._total = total
        self._set_state("executing", mode="live" if live else "preview", total=total,
                        message="running the task" if live else "previewing the plan")
        per_step_max = max(60, int(step_seconds * POLICY_RATE_HZ) + 30)
        prev_key: str | None = None
        for i, st in enumerate(steps):
            self._node(f"step-{i}", "active")   # green border on the running node
            self._emit("step", index=i, total=total, key=st["key"], object=st["object"],
                       label=st["label"], emoji=st["emoji"], prompt=st["prompt"], status="active")
            await self._announce_swap(i, prev_key, st)
            prev_key = st["key"]
            await self._run_one_step(i, total, st, step_seconds, per_step_max, live=live)
            self._node(f"step-{i}", "done")
            self._emit("step", index=i, total=total, key=st["key"], status="done")

            # confirm with the camera, moving the arm out of the way first
            self._node(f"confirm-{i}", "active")
            ok, conf, note = await self._confirm(st, i, live=live)
            if not ok and live:
                self._log(f"step {i + 1} not confirmed ({note}), retrying once")
                await self._run_one_step(i, total, st, step_seconds, per_step_max, live=live)
                ok, conf, note = await self._confirm(st, i, live=live)
            self._node(f"confirm-{i}", "done" if ok else "failed", confidence=conf, note=note)
            self._emit("confirm", index=i, ok=ok, confidence=conf, note=note)

    async def _run_one_step(self, i: int, total: int, st: dict[str, Any],
                            step_seconds: float, per_step_max: int, *, live: bool) -> bool:
        """Drive one object. Live = a fresh policy run primed with this step's
        prompt (the serve hot-swaps to the matching adapter); preview = a timed
        window. Per-step runs leave the arm free between steps for confirmation."""
        if not live:
            await self._run_step_window(i, total, step_seconds, live=False)
            return True
        self.app["policy_prompt"] = st["prompt"]
        if not await self._start_policy(per_step_max):
            self._log("could not start the policy run, previewing this step")
            await self._run_step_window(i, total, step_seconds, live=False)
            return False
        try:
            await self._run_step_window(i, total, step_seconds, live=True)
        finally:
            await self._stop_policy()
        return True

    async def _confirm(self, step: dict[str, Any], index: int, *,
                       live: bool) -> tuple[bool, float | None, str]:
        """Move the arm clear, grab a base-cam frame, and ask the vision model
        whether the object actually landed. Returns (ok, confidence, note)."""
        if not self._use_images:
            return True, None, "no-image mode, marking complete"
        if not do_inference.available():
            return True, None, "no vision model, assumed complete"
        frac = (index + 1) / max(1, self._total)
        for attempt in range(2):
            await self._clear_view(live=live)
            frame = await self._observe(frac, f"Checking: {step.get('object')}")
            if frame is None:
                return True, None, "no camera, assumed complete"
            try:
                res = await do_inference.confirm_action(
                    self._session, frame, step, model=self._model)
            except Exception as exc:  # noqa: BLE001
                return True, None, f"confirm skipped ({exc})"
            if res.get("arm_blocking") and attempt == 0:
                self._log(f"step {index + 1}: arm blocking the view, moving it aside and re-checking")
                continue
            ok = bool(res["done"]) and res["confidence"] >= self._confirm_threshold
            note = res["note"] or ("looks placed" if ok else "not on the box yet")
            return ok, res["confidence"], note
        return False, 0.0, "could not get a clear view of the box"

    async def _clear_view(self, *, live: bool) -> None:
        """Move the arm to home so it isn't occluding the camera, then let the
        frame settle. Best-effort; only moves the arm when actually driving it."""
        if live:
            try:
                async with self._session.post(
                    f"{self.self_base}/v1/teleop/home",
                    timeout=aiohttp.ClientTimeout(total=15),
                ):
                    pass
            except Exception as exc:  # noqa: BLE001
                self._log(f"clear-view home move failed: {exc}")
        await asyncio.sleep(0.6 if live else 0.2)

    async def _announce_swap(self, index: int, prev_key: str | None, st: dict[str, Any]) -> None:
        self._emit("swap", index=index, from_key=prev_key, to_key=st["key"],
                   from_label=(SKILL_BY_KEY.get(prev_key, {}).get("label") if prev_key else None),
                   to_label=st["label"], repo=st.get("repo", ""))
        await asyncio.sleep(0.5)

    async def _run_step_window(self, index: int, total: int, step_seconds: float, *, live: bool) -> None:
        """Hold on the current step for its time budget, emitting progress.
        Progress liveness comes from the eval client's heartbeat when live."""
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            pct = min(100.0, 100.0 * elapsed / step_seconds)
            note = ""
            if live:
                proc = self.app.get("eval_process")
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(
                        f"policy run exited early (code {proc.poll()}) during step {index + 1}")
                hb = self.app.get("policy_heartbeat") or {}
                idx = hb.get("last_action_idx")
                note = f"action {idx}" if idx is not None else "driving arm"
            self._emit("progress", index=index, total=total, pct=round(pct, 1), note=note, live=live)
            if pct >= 100.0:
                return
            await asyncio.sleep(0.4)

    # -- daemon self-calls ---------------------------------------------------

    async def _observe(self, frac: float, caption: str) -> bytes | None:
        """Capture a frame, push it to the page as an observed image, return it."""
        frame = await self._capture_frame(frac)
        if frame is not None:
            url = "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")
            self._emit("image", url=url, caption=caption)
        return frame

    async def _capture_frame(self, frac: float = 0.0) -> bytes | None:
        supervisor = self.app.get("supervisor")
        backend = getattr(supervisor, "_backend", None) if supervisor else None
        fast = getattr(backend, "camera_jpeg", None)
        if callable(fast):
            try:
                b = await asyncio.to_thread(fast, "base")
                if b:
                    return b
            except Exception:  # noqa: BLE001
                pass
        # Fall back to recorded sample footage when there's no live camera.
        if getattr(self, "_use_images", True) and getattr(self, "_sample_id", None):
            try:
                return await asyncio.to_thread(samples.frame_at, self._sample_id, frac)
            except Exception:  # noqa: BLE001
                return None
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
        # One multimodal model does both planning and image recognition. All are
        # OpenAI (called directly with OPENAI_API_KEY); llama-4-maverick is a
        # DigitalOcean open-weights fallback.
        "models": ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-4o", "gpt-4o-mini",
                   "gpt-4.1", "llama-4-maverick"],
        "default_model": do_inference.DEFAULT_THINKING_MODEL,
        "abilities": abilities.list_abilities(),
    }
