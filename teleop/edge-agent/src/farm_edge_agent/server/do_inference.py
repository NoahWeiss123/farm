"""DigitalOcean serverless-inference client — scene vision + task planning.

The ``/user`` consumer page asks a high-level task ("place the items on the
box"); this module turns that into an ordered, skill-routed plan:

  1. ``detect_objects`` — a vision model looks at the live base-camera frame and
     reports which of the trainable objects (bottle / bear / duck / hat) are
     actually on the table.
  2. ``plan_task`` — a reasoning model (GPT-5.5) decomposes the task into ordered
     per-object steps, each bound to one of the on-robot *skills* (the per-task
     LoRAs that hot-swap on top of the frozen FFT-56k base).

Everything runs **server-side in the daemon** so the DigitalOcean key never
reaches the browser. The endpoint is OpenAI-compatible
(``https://inference.do-ai.run/v1/chat/completions``), so the wire format is the
standard chat-completions shape with ``Authorization: Bearer <model-access-key>``.

The key is read from the ``DigitalOcean`` environment variable (the name used in
this repo's ``.env``); if the daemon wasn't started with it exported, we fall
back to parsing ``<repo>/.env`` directly. Alternate names
(``DIGITALOCEAN_INFERENCE_KEY``, ``DO_INFERENCE_KEY``) are also accepted.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp

log = logging.getLogger("farm.do")

BASE_URL = "https://inference.do-ai.run/v1"
# NB: the OpenAI/Anthropic slugs (gpt-5.5, claude-*) are listed by /v1/models but
# 403 on this account's subscription tier — only open-weights models are callable.
# llama-4-maverick is the strongest available and is multimodal, so it serves as
# both the default "thinking" model and the default scene/vision model. Both are
# user-overridable from the /user page (see get_agent_config in app.py).
DEFAULT_THINKING_MODEL = "llama-4-maverick"
DEFAULT_VISION_MODEL = "llama-4-maverick"

# Env var names that may hold the DO model-access key, most-specific first. The
# bare ``DigitalOcean`` is what this repo's .env uses.
_KEY_NAMES = ("DigitalOcean", "DIGITALOCEAN_INFERENCE_KEY", "DO_INFERENCE_KEY",
              "DIGITALOCEAN_API_KEY", "DO_API_KEY")
# server → farm_edge_agent → src → edge-agent → teleop → repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]

_key_cache: dict[str, str | None] = {}


def _parse_env_file(path: Path, name: str) -> str | None:
    """Pull a single ``NAME=value`` out of a .env file without importing dotenv.
    Tolerates ``export `` prefixes and surrounding single/double quotes."""
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, val = line.partition("=")
            if sep and key.strip() == name:
                val = val.strip()
                if val and val[0] in "\"'":
                    # quoted value: take up to the matching close quote, so a
                    # trailing inline comment never leaks into the key.
                    end = val.find(val[0], 1)
                    return (val[1:end] if end != -1 else val[1:]) or None
                # unquoted value: drop a trailing " # comment" if present.
                hashpos = val.find(" #")
                if hashpos != -1:
                    val = val[:hashpos].rstrip()
                return val or None
    except OSError:
        return None
    return None


def api_key() -> str | None:
    """The DO model-access key, or None if it can't be found. Cached.

    Checks the process environment first (any of ``_KEY_NAMES``), then falls
    back to parsing ``<repo>/.env`` — the daemon isn't started with .env
    exported, so the file fallback is the common path.
    """
    if "key" in _key_cache:
        return _key_cache["key"]
    import os

    found: str | None = None
    for name in _KEY_NAMES:
        v = os.environ.get(name)
        if v and v.strip():
            found = v.strip()
            break
    if found is None:
        env_path = _REPO_ROOT / ".env"
        for name in _KEY_NAMES:
            v = _parse_env_file(env_path, name)
            if v:
                found = v
                break
    _key_cache["key"] = found
    if found is None:
        log.warning("no DigitalOcean key found (looked for %s in env + %s/.env)",
                    "/".join(_KEY_NAMES), _REPO_ROOT)
    return found


def available() -> bool:
    return api_key() is not None


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def _extract_json(text: str) -> Any | None:
    """Best-effort: pull the first JSON object/array out of model output that
    may wrap it in prose or a ```json fence."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    # Also try the first balanced {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:i + 1])
                        break
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


# ── raw chat-completions ────────────────────────────────────────────────────


async def stream_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, Any]],
    *,
    on_delta: Callable[[str], Any] | None = None,
    timeout: float = 120.0,
    **params: Any,
) -> str:
    """Stream a chat completion, forwarding each visible content delta to
    ``on_delta`` (for the live "thinking" panel) and returning the full text.

    Params are kept minimal (no temperature) because the GPT-5.x reasoning
    models reject non-default sampling knobs. Raises on HTTP/auth errors.
    """
    payload = {"model": model, "messages": messages, "stream": True, **params}
    full: list[str] = []
    async with session.post(
        f"{BASE_URL}/chat/completions",
        headers=_headers(),
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            # Log the upstream body server-side only; never forward the provider's
            # raw response (tier/billing/internal hints) to the unauthenticated page.
            body = (await resp.text())[:500]
            log.warning("DO %s HTTP %s: %s", model, resp.status, body)
            raise RuntimeError(f"DO inference failed for {model} (HTTP {resp.status})")
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            try:
                delta = chunk["choices"][0].get("delta", {})
            except (KeyError, IndexError):
                continue
            piece = delta.get("content")
            if piece:
                full.append(piece)
                if on_delta is not None:
                    on_delta(piece)
    return "".join(full)


# ── high-level steps ────────────────────────────────────────────────────────


async def detect_objects(
    session: aiohttp.ClientSession,
    image_jpeg: bytes,
    candidates: list[dict[str, str]],
    *,
    model: str = DEFAULT_VISION_MODEL,
    on_delta: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Vision pass over the live base-camera frame.

    ``candidates`` is the skill catalogue ``[{key, object, label}]``. Returns
    ``{"present": [skill_key, ...], "objects": [{key, object, present,
    confidence, note}], "summary": str, "raw": str}`` ordered the way the items
    should be picked (front-to-back / nearest-first is fine — the planner
    re-orders anyway).
    """
    catalogue = "\n".join(
        f"  - key={c['key']!r}  object={c['object']!r}" for c in candidates
    )
    b64 = base64.b64encode(image_jpeg).decode("ascii")
    sys_prompt = (
        "You are the perception module of a tabletop pick-and-place robot. "
        "You are shown one camera frame of the workspace. Identify which of the "
        "KNOWN OBJECTS are physically present on the table in front of the arm. "
        "Briefly narrate what you see, then output a fenced ```json block with "
        'shape {"summary": "<one short sentence>", "present": ["<key>", ...], '
        '"objects": [{"key","object","present": true|false,"confidence": 0..1,"note": "<short>"}]}. '
        "Only use keys from the known-objects list. Order 'present' by pick "
        "convenience (nearest/least-occluded first)."
    )
    user_content = [
        {"type": "text", "text": f"KNOWN OBJECTS:\n{catalogue}\n\nWhat is on the table?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]
    text = await stream_chat(
        session, model,
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_content}],
        on_delta=on_delta,
    )
    parsed = _extract_json(text) or {}
    valid = {c["key"] for c in candidates}
    present = [k for k in (parsed.get("present") or []) if k in valid]
    objects = [o for o in (parsed.get("objects") or []) if o.get("key") in valid]
    # Fallback: if the model gave objects[] but no present[], derive it.
    if not present and objects:
        present = [o["key"] for o in objects if o.get("present")]
    return {
        "present": present,
        "objects": objects,
        "summary": (parsed.get("summary") or "").strip(),
        "raw": text,
    }


async def plan_task(
    session: aiohttp.ClientSession,
    task: str,
    present: list[dict[str, str]],
    skills: list[dict[str, str]],
    *,
    model: str = DEFAULT_THINKING_MODEL,
    on_delta: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Decompose ``task`` into ordered, skill-bound steps.

    ``present`` = the detected objects ``[{key, object}]``; ``skills`` = the
    enabled skill catalogue ``[{key, object, label, prompt}]``. The model thinks
    out loud (streamed to ``on_delta``) and ends with a ```json plan. Returns
    ``{"steps": [{key, object, prompt, rationale}], "summary": str, "raw": str}``
    where each step's ``prompt`` is the exact policy task string for that skill.
    """
    skill_lines = "\n".join(
        f"  - key={s['key']!r}  object={s['object']!r}  policy_prompt={s['prompt']!r}"
        for s in skills
    )
    present_lines = ", ".join(p["object"] for p in present) or "(none detected)"
    sys_prompt = (
        "You are the planner for a UF850 robot arm that picks up tabletop "
        "objects and places them on a box. The arm runs a frozen base policy "
        "(full fine-tune) and hot-swaps a small per-object LoRA 'skill' for each "
        "pick. You break a high-level request into an ORDERED sequence of "
        "single-object pick-and-place steps, one per object that should be "
        "moved, each bound to exactly one available skill.\n\n"
        "Rules:\n"
        "- Only use skills from the AVAILABLE SKILLS list (match by object).\n"
        "- One step per object instance to move. Order them sensibly "
        "(least-occluded / topmost first to avoid disturbing others).\n"
        "- Each step's 'prompt' MUST be the skill's exact policy_prompt.\n"
        "- If the request can't be done with the available skills, still plan "
        "the steps you can and note the gap.\n\n"
        "Think briefly out loud about the scene and ordering, then output a "
        'fenced ```json block: {"summary": "<one sentence>", "steps": '
        '[{"key","object","prompt","rationale"}]}.'
    )
    user_prompt = (
        f"REQUEST: {task}\n\n"
        f"DETECTED ON TABLE: {present_lines}\n\n"
        f"AVAILABLE SKILLS:\n{skill_lines}\n\n"
        "Produce the ordered plan."
    )
    text = await stream_chat(
        session, model,
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_prompt}],
        on_delta=on_delta,
    )
    parsed = _extract_json(text) or {}
    by_key = {s["key"]: s for s in skills}
    steps: list[dict[str, Any]] = []
    for raw_step in parsed.get("steps") or []:
        key = raw_step.get("key")
        skill = by_key.get(key)
        if skill is None:
            continue
        steps.append({
            "key": key,
            "object": skill["object"],
            "prompt": skill["prompt"],  # authoritative: the trained policy string
            "rationale": str(raw_step.get("rationale") or "").strip(),
        })
    return {
        "steps": steps,
        "summary": (parsed.get("summary") or "").strip(),
        "raw": text,
    }
