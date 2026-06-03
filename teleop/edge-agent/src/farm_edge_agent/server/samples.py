"""Sample footage: use a recorded episode's base-camera frames as a stand-in
camera, so the consumer agent runs real vision detection + confirmation offline,
as if the recorded video were its own live feed.

``frame_at(sample_id, frac)`` returns the JPEG ``frac`` of the way through the
episode (0.0 = first frame, 1.0 = last). The orchestrator pulls an early frame to
perceive the scene and progressively later frames to confirm each step, so as the
task advances the confirmation sees more of it done.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# server -> farm_edge_agent -> src -> edge-agent -> teleop -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
DATASETS_DIR = _REPO_ROOT / "datasets"
_EP_RE = re.compile(r"^episode_[A-Za-z0-9_\-]+$")

_cache: dict[str, Any] = {"at": 0.0, "val": []}


def _base_dirs() -> list[Path]:
    """Episode base-camera dirs, looked up at the one or two nesting levels the
    datasets use (datasets/<set>/<episode>/ and datasets/<set>/<sub>/<episode>/)."""
    out: list[Path] = []
    if not DATASETS_DIR.is_dir():
        return out
    for pat in ("*/episode_*/cameras/base", "*/*/episode_*/cameras/base"):
        out.extend(p for p in DATASETS_DIR.glob(pat) if p.is_dir())
    return out


def list_samples(limit: int = 30) -> list[dict[str, Any]]:
    """Recorded episodes usable as sample footage, longest first. Cached 30 s."""
    now = time.monotonic()
    if _cache["at"] and now - _cache["at"] < 30.0:
        return _cache["val"]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for base in _base_dirs():
        ep = base.parent.parent
        if not _EP_RE.match(ep.name):
            continue
        try:
            rel = str(ep.relative_to(DATASETS_DIR))
        except ValueError:
            continue
        if rel in seen:
            continue
        n = sum(1 for _ in base.glob("*.jpg"))
        if n == 0:
            continue
        seen.add(rel)
        desc = ""
        meta = ep / "meta.json"
        if meta.is_file():
            try:
                desc = (json.loads(meta.read_text()).get("description") or "").strip()
            except Exception:  # noqa: BLE001
                desc = ""
        out.append({"id": rel, "description": desc or ep.name, "n_frames": n})
    out.sort(key=lambda s: s["n_frames"], reverse=True)
    out = out[:limit]
    _cache.update(at=now, val=out)
    return out


def frame_at(sample_id: str, frac: float) -> bytes | None:
    """The base-cam JPEG ``frac`` of the way through the episode, or None."""
    if not sample_id:
        return None
    ep = (DATASETS_DIR / sample_id).resolve()
    try:
        ep.relative_to(DATASETS_DIR.resolve())
    except ValueError:
        return None
    base = ep / "cameras" / "base"
    if not base.is_dir():
        return None
    frames = sorted(base.glob("*.jpg"))
    if not frames:
        return None
    frac = max(0.0, min(1.0, frac))
    idx = min(len(frames) - 1, int(round(frac * (len(frames) - 1))))
    try:
        return frames[idx].read_bytes()
    except OSError:
        return None
