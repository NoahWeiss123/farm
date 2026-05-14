"""Skill library — registered skills plus the executor that dispatches them.

A skill is a Python function that takes a `SkillContext` (current world
state + driver) and a dict of arguments parsed from the plan node, and
returns a list of action chunks. Each chunk goes through the safety
enforcer before reaching the driver.

Adding a new skill: define it here (or in a sibling module), call
``register("name", fn, schema=...)``, and reference it by name from the
planner's output.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from farm_edge_agent.drivers.base import Driver
from farm_edge_agent.run_loop import ExecResult, PlanNode

APPROACH_HEIGHT_MM = 120.0
DROP_HEIGHT_MM = 60.0
STACK_HEIGHT_MM = 40.0
DEFAULT_VELOCITY_CAP = 100.0


@dataclass
class SkillContext:
    """Everything a skill needs that isn't its arguments."""

    driver: Driver
    world_props: dict[str, tuple[float, float, float]]
    run_id: str


@dataclass
class SkillSpec:
    """Schema entry for a single registered skill."""

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    fn: Callable[[SkillContext, dict[str, Any]], list[dict[str, Any]]] | None = None


_REGISTRY: dict[str, SkillSpec] = {}


def register(spec: SkillSpec) -> None:
    _REGISTRY[spec.name] = spec


def get(name: str) -> SkillSpec | None:
    return _REGISTRY.get(name)


def all_specs() -> list[SkillSpec]:
    return list(_REGISTRY.values())


# ── helpers ──────────────────────────────────────────────────────────────────


def _meters_to_mm(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    return (pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0)


def _resolve_prop(
    name: str, props: Mapping[str, tuple[float, float, float]]
) -> tuple[str, tuple[float, float, float]] | None:
    canon = re.sub(r"[\s_-]+", " ", name).strip().lower()
    direct = canon.replace(" ", "_")
    if direct in props:
        return direct, props[direct]
    tokens = canon.split()
    for prop_id, pos in props.items():
        prop_tokens = prop_id.lower().split("_")
        if all(any(t in pt for pt in prop_tokens) for t in tokens):
            return prop_id, pos
    return None


def _approach_sequence(
    source: tuple[str, tuple[float, float, float]],
    target_pos_mm: tuple[float, float, float],
    *,
    approach_height_mm: float = APPROACH_HEIGHT_MM,
    drop_height_mm: float = DROP_HEIGHT_MM,
    velocity_cap: float = DEFAULT_VELOCITY_CAP,
    note: str | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical approach-grasp-lift-place sequence."""
    src_id, src_pos = source
    sx, sy, sz = _meters_to_mm(src_pos)
    tx, ty, tz = target_pos_mm
    rpy = (math.pi, 0.0, 0.0)
    chunks: list[dict[str, Any]] = []

    def waypoint(x: float, y: float, z: float, label: str) -> None:
        chunks.append(
            {
                "type": "action_chunk",
                "action_space": "tcp_xyzrpy_mm",
                "action": [x, y, z, rpy[0], rpy[1], rpy[2]],
                "velocity_cap": velocity_cap,
                "label": label,
            }
        )

    def grip(state: str, label: str) -> None:
        chunks.append(
            {
                "type": "action_chunk",
                "action_space": "gripper",
                "action": [state],
                "label": label,
            }
        )

    if note:
        chunks.append({"type": "critic_note", "text": note})
    waypoint(sx, sy, sz + approach_height_mm, f"above_{src_id}")
    grip("open", "open_gripper")
    waypoint(sx, sy, sz, f"at_{src_id}")
    grip("closed", f"grasp_{src_id}")
    waypoint(sx, sy, sz + approach_height_mm, f"lift_from_{src_id}")
    waypoint(tx, ty, tz + approach_height_mm, "above_target")
    waypoint(tx, ty, tz + drop_height_mm, "drop_target")
    grip("open", "release")
    waypoint(tx, ty, tz + approach_height_mm, "retract")
    return chunks


# ── skill implementations ───────────────────────────────────────────────────


def _pick_and_place(ctx: SkillContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    source = str(args.get("source") or args.get("from") or "")
    target = str(args.get("target") or args.get("to") or "")
    src = _resolve_prop(source, ctx.world_props)
    tgt = _resolve_prop(target, ctx.world_props)
    if src is None:
        raise ValueError(f"unknown source prop: {source!r}")
    if tgt is None:
        raise ValueError(f"unknown target prop: {target!r}")
    return _approach_sequence(
        source=src,
        target_pos_mm=_meters_to_mm(tgt[1]),
        note=(
            f"pick_and_place: {src[0]} → on top of {tgt[0]}"
        ),
    )


def _stack(ctx: SkillContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    top = str(args.get("top") or args.get("source") or "")
    bottom = str(args.get("bottom") or args.get("target") or "")
    top_res = _resolve_prop(top, ctx.world_props)
    bottom_res = _resolve_prop(bottom, ctx.world_props)
    if top_res is None:
        raise ValueError(f"unknown top prop: {top!r}")
    if bottom_res is None:
        raise ValueError(f"unknown bottom prop: {bottom!r}")
    # Drop right onto the bottom block's top face.
    return _approach_sequence(
        source=top_res,
        target_pos_mm=_meters_to_mm(bottom_res[1]),
        drop_height_mm=STACK_HEIGHT_MM,
        note=f"stack: {top_res[0]} on {bottom_res[0]}",
    )


def _go_to(ctx: SkillContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Plain TCP move. Args: x, y, z (mm) plus optional rpy (radians)."""
    x = float(args.get("x", 0))
    y = float(args.get("y", -700))
    z = float(args.get("z", 400))
    rx = float(args.get("rx", math.pi))
    ry = float(args.get("ry", 0))
    rz = float(args.get("rz", 0))
    return [
        {"type": "critic_note", "text": f"go_to ({x:.0f}, {y:.0f}, {z:.0f})mm"},
        {
            "type": "action_chunk",
            "action_space": "tcp_xyzrpy_mm",
            "action": [x, y, z, rx, ry, rz],
            "velocity_cap": DEFAULT_VELOCITY_CAP,
            "label": "go_to",
        },
    ]


def _home(ctx: SkillContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Return arm to home pose."""
    # Home pose mirrors SimDriver.HOME_POSE
    return [
        {"type": "critic_note", "text": "return to home"},
        {
            "type": "action_chunk",
            "action_space": "tcp_xyzrpy_mm",
            "action": [0.0, -668.0, 396.0, math.pi, 0.0, 0.0],
            "velocity_cap": DEFAULT_VELOCITY_CAP,
            "label": "home",
        },
        {
            "type": "action_chunk",
            "action_space": "gripper",
            "action": ["open"],
            "label": "open_gripper",
        },
    ]


register(SkillSpec(
    name="pick_and_place",
    description="Pick up `source` prop and place it on top of `target` prop.",
    parameters={
        "source": "string — id of the prop to grasp (e.g. 'red_block')",
        "target": "string — id of the prop to place onto (e.g. 'cup')",
    },
    fn=_pick_and_place,
))
register(SkillSpec(
    name="stack",
    description="Stack `top` on `bottom` (synonym for pick_and_place with a tighter drop height).",
    parameters={
        "top": "string — id of the prop to grasp",
        "bottom": "string — id of the prop to stack on",
    },
    fn=_stack,
))
register(SkillSpec(
    name="go_to",
    description="Move the TCP to a Cartesian pose. Coordinates in millimetres, RPY in radians.",
    parameters={
        "x": "float — TCP x in mm",
        "y": "float — TCP y in mm",
        "z": "float — TCP z in mm",
        "rx": "float (optional) — TCP roll",
        "ry": "float (optional) — TCP pitch",
        "rz": "float (optional) — TCP yaw",
    },
    fn=_go_to,
))
register(SkillSpec(
    name="home",
    description="Return the arm to its home pose and open the gripper.",
    parameters={},
    fn=_home,
))


# ── dispatching executor ────────────────────────────────────────────────────


class WorldStateProvider:
    """Pluggable source of (prop_id → 3D meter position) the executor reads
    fresh before every node so skills always see the latest physics state."""

    def world_state(self) -> Mapping[str, tuple[float, float, float]]:
        return {}


class StaticWorldState(WorldStateProvider):
    def __init__(self, props: Mapping[str, tuple[float, float, float]]) -> None:
        self._props = dict(props)

    def world_state(self) -> Mapping[str, tuple[float, float, float]]:
        return self._props


class LiveSimWorldState(WorldStateProvider):
    """Pulls live prop positions out of the SimDriver each call."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    def world_state(self) -> Mapping[str, tuple[float, float, float]]:
        snap = self._driver.snapshot()
        out: dict[str, tuple[float, float, float]] = {}
        for pid, p in snap.get("props", {}).items():
            pos = p.get("pos") or [0, 0, 0]
            out[pid] = (float(pos[0]), float(pos[1]), float(pos[2]))
        return out


@dataclass
class SkillCall:
    """One step in a plan — name + JSON args."""

    skill: str
    args: dict[str, Any] = field(default_factory=dict)
    node_id: str = ""


class SkillExecutor:
    """Executes plan nodes whose ``instruction`` is either:

    - A JSON object: ``{"skill": "...", "args": {...}}``
    - A JSON array: a list of such objects (multi-step within one node)
    - An English instruction (fallback): parsed for "pick X place on Y".
    """

    def __init__(self, world: WorldStateProvider) -> None:
        self._world = world

    def execute(self, node: PlanNode, driver: Driver, run_id: str) -> ExecResult:
        calls = self._parse(node.instruction)
        if calls is None:
            return ExecResult(
                ok=False,
                error=f"executor could not parse node instruction: {node.instruction!r}",
            )
        props = dict(self._world.world_state())
        ctx = SkillContext(driver=driver, world_props=props, run_id=run_id)
        chunks: list[dict[str, Any]] = []
        for call in calls:
            spec = get(call.skill)
            if spec is None or spec.fn is None:
                return ExecResult(ok=False, error=f"unknown skill: {call.skill!r}")
            try:
                produced = spec.fn(ctx, dict(call.args))
            except Exception as e:
                return ExecResult(
                    ok=False,
                    error=f"skill {call.skill!r} failed: {type(e).__name__}: {e}",
                )
            chunks.extend(produced)
            # Refresh world state after each skill — props may have moved.
            ctx = SkillContext(
                driver=driver,
                world_props=dict(self._world.world_state()),
                run_id=run_id,
            )
        return ExecResult(ok=True, chunks=chunks)

    @staticmethod
    def _parse(instruction: str) -> list[SkillCall] | None:
        import json

        text = instruction.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(obj, dict) and "skill" in obj:
                return [SkillCall(skill=str(obj["skill"]), args=dict(obj.get("args") or {}))]
            if isinstance(obj, list):
                out: list[SkillCall] = []
                for entry in obj:
                    if isinstance(entry, dict) and "skill" in entry:
                        out.append(
                            SkillCall(
                                skill=str(entry["skill"]),
                                args=dict(entry.get("args") or {}),
                            )
                        )
                return out or None
        # English fallback: try the pick-and-place pattern.
        from farm_edge_agent.skills.executor import _parse_pick_place  # type: ignore

        parsed = _parse_pick_place(instruction)
        if parsed is None:
            return None
        return [SkillCall(skill="pick_and_place", args={"source": parsed[0], "target": parsed[1]})]


__all__ = [
    "LiveSimWorldState",
    "SkillCall",
    "SkillContext",
    "SkillExecutor",
    "SkillSpec",
    "StaticWorldState",
    "WorldStateProvider",
    "all_specs",
    "get",
    "register",
]
