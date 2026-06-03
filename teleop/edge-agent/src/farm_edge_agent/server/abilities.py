"""Saved "Abilities" — reusable agent workflows.

An Ability is a generated plan (an ordered list of skill steps) saved to disk so
it can be re-run later WITHOUT redoing vision detection + LLM planning. That makes
a proven workflow reproducible and fast: load the ability, drive the same skill
sequence, confirm each step. Stored as JSON under ``<repo>/abilities/``.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# server -> farm_edge_agent -> src -> edge-agent -> teleop -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
ABILITIES_DIR = _REPO_ROOT / "abilities"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:48] or "ability"


def _path(aid: str) -> Path | None:
    """Resolve an ability id to a file path, refusing traversal."""
    if not _ID_RE.match(aid or ""):
        return None
    p = (ABILITIES_DIR / f"{aid}.json").resolve()
    try:
        p.relative_to(ABILITIES_DIR.resolve())
    except ValueError:
        return None
    return p


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def list_abilities() -> list[dict[str, Any]]:
    """Lightweight summaries of every saved ability, newest first."""
    out: list[dict[str, Any]] = []
    if ABILITIES_DIR.is_dir():
        for p in sorted(ABILITIES_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            steps = d.get("steps") or []
            out.append({
                "id": d.get("id", p.stem),
                "name": d.get("name", p.stem),
                "task": d.get("task", ""),
                "base_model": d.get("base_model"),
                "n_steps": len(steps),
                "objects": d.get("objects") or [s.get("key") for s in steps],
                "created_at": d.get("created_at"),
            })
    out.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    return out


def get_ability(aid: str) -> dict[str, Any] | None:
    p = _path(aid)
    if p is None or not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def save_ability(data: dict[str, Any]) -> dict[str, Any]:
    """Persist an ability. ``data`` needs ``name`` (or ``task``) and ``steps``;
    re-saving the same slug overwrites. Returns the stored record."""
    name = str(data.get("name") or data.get("task") or "ability").strip()
    aid = str(data.get("id") or "").strip() or _slug(name)
    if not _ID_RE.match(aid):
        aid = _slug(name)
    steps: list[dict[str, Any]] = []
    for s in data.get("steps") or []:
        if not isinstance(s, dict) or "key" not in s:
            continue
        steps.append({
            "key": s["key"],
            "object": s.get("object", ""),
            "label": s.get("label", s["key"]),
            "emoji": s.get("emoji", ""),
            "prompt": s.get("prompt", ""),
            "repo": s.get("repo", ""),
            "rationale": s.get("rationale", ""),
        })
    if not steps:
        raise ValueError("an ability needs at least one step")
    rec = {
        "id": aid,
        "name": name,
        "task": str(data.get("task") or ""),
        "summary": str(data.get("summary") or ""),
        "base_model": data.get("base_model") or "fft_hotswap",
        "objects": [s["key"] for s in steps],
        "steps": steps,
        "created_at": data.get("created_at") or _now_iso(),
    }
    p = _path(aid)
    if p is None:
        raise ValueError("invalid ability id")
    ABILITIES_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2))
    return rec


def delete_ability(aid: str) -> bool:
    p = _path(aid)
    if p is None or not p.is_file():
        return False
    p.unlink()
    return True
