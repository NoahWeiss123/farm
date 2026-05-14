"""GPT-driven hierarchical task decomposer.

Sends the user prompt + current scene + the registered skill catalog to
OpenAI Chat Completions, parses the returned JSON plan into a list of
``PlanNode``s whose instruction is a JSON-encoded ``{skill, args}`` object
the SkillExecutor knows how to dispatch.

Plan caching (Layer-1 skill compiler) lives here as well — we keep an
on-disk SQLite of ``(task, scene_hash) → plan_json`` so the second run of
the same prompt skips the LLM call entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from farm_edge_agent.run_loop import Plan, PlanNode
from farm_edge_agent.skills.library import all_specs

log = logging.getLogger("farm.planner")

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
CACHE_PATH = Path.home() / ".farm" / "plan_cache.db"


@dataclass
class PlanCacheHit:
    plan: Plan
    age_s: float


class PlanCache:
    """Tiny SQLite cache keyed by (task, scene_hash, model)."""

    def __init__(self, path: Path = CACHE_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS plans ("
            "task TEXT NOT NULL,"
            "scene_hash TEXT NOT NULL,"
            "model TEXT NOT NULL,"
            "plan_json TEXT NOT NULL,"
            "created_at REAL NOT NULL,"
            "PRIMARY KEY (task, scene_hash, model))"
        )
        self._conn.commit()

    def get(self, task: str, scene_hash: str, model: str) -> PlanCacheHit | None:
        cur = self._conn.execute(
            "SELECT plan_json, created_at FROM plans "
            "WHERE task=? AND scene_hash=? AND model=?",
            (task, scene_hash, model),
        )
        row = cur.fetchone()
        if row is None:
            return None
        raw = json.loads(row[0])
        return PlanCacheHit(plan=_plan_from_json(raw), age_s=time.time() - float(row[1]))

    def put(self, task: str, scene_hash: str, model: str, plan: Plan) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO plans VALUES (?,?,?,?,?)",
            (task, scene_hash, model, json.dumps(_plan_to_json(plan)), time.time()),
        )
        self._conn.commit()


def _plan_to_json(plan: Plan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "reasoning": plan.reasoning,
        "nodes": [
            {"id": n.id, "instruction": n.instruction, "backend": n.backend}
            for n in plan.nodes
        ],
    }


def _plan_from_json(d: Mapping[str, Any]) -> Plan:
    return Plan(
        plan_id=str(d.get("plan_id", "")),
        reasoning=str(d.get("reasoning", "")),
        nodes=[
            PlanNode(
                id=str(n.get("id", f"n{i+1}")),
                instruction=str(n.get("instruction", "")),
                backend=str(n.get("backend", "sim")),
            )
            for i, n in enumerate(d.get("nodes", []))
        ],
    )


def _scene_hash(scene: Mapping[str, Any]) -> str:
    """Hash only the structural parts of the scene — prop ids + colors + shapes.

    Live positions are deliberately dropped: once a user types "stack blue on
    green", the same plan should apply across small position drift, otherwise
    the second run after blue moved would always re-call the LLM.
    """
    skeleton = {
        "scene_name": scene.get("scene_name"),
        "props": sorted(
            (
                {
                    "id": p.get("id"),
                    "shape": p.get("shape"),
                    "rgba": p.get("rgba"),
                }
                for p in scene.get("props", [])
            ),
            key=lambda p: str(p.get("id")),
        ),
    }
    canonical = json.dumps(skeleton, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _build_system_prompt() -> str:
    skills = []
    for spec in all_specs():
        params = ", ".join(f"{k}: {v}" for k, v in spec.parameters.items())
        skills.append(
            f"- {spec.name}({params}): {spec.description}"
        )
    skill_table = "\n".join(skills)
    return (
        "You are the task planner for a UFactory 850 6-DOF arm with a parallel-jaw "
        "gripper, operating in a tabletop sim. Decompose the user's natural-language "
        "task into a JSON plan that calls one or more skills from the library below. "
        "Use prop ids exactly as they appear in the scene.\n\n"
        f"Available skills:\n{skill_table}\n\n"
        'Respond with strict JSON of shape: {"plan_id": "plan_xxx", "reasoning": '
        '"...", "nodes": [{"id": "n1", "skill": "pick_and_place", "args": '
        '{"source": "red_block", "target": "cup"}}, ...]}. Each node\'s `id` must '
        "be unique within the plan. Do not invent prop ids — only use ones from "
        "the scene description."
    )


@dataclass
class GptPlanner:
    """OpenAI-driven planner. Reads OPENAI_API_KEY from env."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    cache: PlanCache | None = None
    scene_provider: Any = None  # callable returning current scene dict
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.cache is None:
            self.cache = PlanCache()

    def plan(self, task: str, *, run_id: str) -> Plan:
        scene = self._current_scene()
        scene_hash = _scene_hash(scene)
        # Cache lookup
        cached = self.cache.get(task, scene_hash, self.model) if self.cache else None
        if cached is not None:
            log.info("plan-cache hit for task=%r age=%.1fs", task, cached.age_s)
            plan = cached.plan
            plan.reasoning = f"{plan.reasoning} (cached, age {cached.age_s:.0f}s)"
            return plan
        if not self.api_key:
            log.warning("no OPENAI_API_KEY; falling back to single-node english plan")
            return Plan(
                plan_id=f"plan_fallback_{hashlib.sha1(task.encode()).hexdigest()[:8]}",
                reasoning="OPENAI_API_KEY missing; using english fallback parser",
                nodes=[PlanNode(id="n1", instruction=task, backend="sim")],
            )
        plan = self._call_openai(task, scene)
        if self.cache is not None:
            self.cache.put(task, scene_hash, self.model, plan)
        return plan

    def _current_scene(self) -> dict[str, Any]:
        if self.scene_provider is None:
            return {}
        try:
            return dict(self.scene_provider())
        except Exception as e:
            log.warning("scene provider failed: %s", e)
            return {}

    def _call_openai(self, task: str, scene: dict[str, Any]) -> Plan:
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _build_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"Scene:\n{json.dumps(scene, indent=2)}\n\n"
                        f"User task: {task}"
                    ),
                },
            ],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"openai chat.completions returned {e.code}: {e.read().decode('utf-8')[:200]}"
            ) from e
        text = payload["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"planner returned invalid JSON: {e}; raw={text[:400]}") from e
        return self._plan_from_response(task, parsed)

    @staticmethod
    def _plan_from_response(task: str, raw: dict[str, Any]) -> Plan:
        plan_id = str(raw.get("plan_id") or f"plan_gpt_{hashlib.sha1(task.encode()).hexdigest()[:8]}")
        reasoning = str(raw.get("reasoning") or "")
        nodes: list[PlanNode] = []
        for i, n in enumerate(raw.get("nodes") or []):
            skill = str(n.get("skill") or "").strip()
            args = n.get("args") or {}
            if not skill:
                continue
            nodes.append(
                PlanNode(
                    id=str(n.get("id") or f"n{i+1}"),
                    instruction=json.dumps({"skill": skill, "args": args}, separators=(",", ":")),
                    backend=str(n.get("backend") or "sim"),
                )
            )
        if not nodes:
            # Some responses use {"plan": [...]} or {"steps": [...]}; tolerate.
            for key in ("plan", "steps"):
                fallback = raw.get(key)
                if isinstance(fallback, list):
                    nodes = [
                        PlanNode(
                            id=str(f.get("id") or f"n{i+1}"),
                            instruction=json.dumps(
                                {"skill": str(f.get("skill")), "args": f.get("args") or {}},
                                separators=(",", ":"),
                            ),
                            backend="sim",
                        )
                        for i, f in enumerate(fallback)
                        if isinstance(f, dict) and f.get("skill")
                    ]
                    break
        if not nodes:
            raise RuntimeError(f"planner produced no usable nodes; raw={raw}")
        return Plan(plan_id=plan_id, reasoning=reasoning, nodes=nodes)


__all__ = ["GptPlanner", "PlanCache"]
