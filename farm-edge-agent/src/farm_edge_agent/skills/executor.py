"""SkillExecutor — generates ActionChunk streams for a plan node.

Phase 3 ships a hand-coded ``PickPlaceExecutor`` that converts a node like
"pick the red block and place it on the cup" into:

    above(source) → descend(source) → grasp → lift →
    above(target) → descend(target) → release → retract

Each phase becomes one ``action_chunk`` event the RunLoop forwards to the
safety enforcer + driver. The executor relies on a ``WorldState`` mapping
of prop ids → 3D positions so the same code drives sim runs (positions
from the SimDriver snapshot) and future real-robot runs (positions from
perception).

The full LLM-driven code-as-policy executor (Phase 9) will replace this
with generated Python; the contract is intentionally narrow so the
RunLoop doesn't need to change.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from farm_edge_agent.drivers.base import Driver
from farm_edge_agent.run_loop import ExecResult, PlanNode

APPROACH_HEIGHT_MM = 120.0
DROP_HEIGHT_MM = 60.0
DEFAULT_VELOCITY_CAP = 100.0


class WorldStateSource(Protocol):
    def world_state(self) -> Mapping[str, tuple[float, float, float]]: ...


@dataclass
class HardcodedWorldState:
    """Static prop positions (in meters) supplied at construction time.

    Useful for tests + the Phase 3 demo where the sim driver already
    knows where everything is.
    """

    props: dict[str, tuple[float, float, float]]

    def world_state(self) -> Mapping[str, tuple[float, float, float]]:
        return self.props


class SkillExecutor(Protocol):
    def execute(
        self,
        node: PlanNode,
        driver: Driver,
        run_id: str,
    ) -> ExecResult: ...


# Synonyms the parser recognizes for the destination in a "place on X" intent.
_PLACEMENT_PREPOSITIONS = ("on top of", "on", "in", "into", "onto")


def _parse_pick_place(instruction: str) -> tuple[str, str] | None:
    """Extract (source_prop, target_prop) from an English instruction.

    Accepts: "pick the red block and place it on the cup",
    "put the red block on the blue block", "place red on cup", etc.
    Returns ``None`` if the instruction doesn't match the pattern.
    """
    lower = instruction.lower()
    # Try the compound "pick A and place on B" pattern first so the "place"
    # regex below doesn't match "place it on B" with subject="it".
    compound = (
        r"pick\s+(?:up\s+)?(?:the\s+)?(.+?)"
        r"\s+(?:and|then|to)\s+(?:place|put|stack|set|drop)\s+(?:it\s+)?"
        r"(?:on top of|onto|on|in|into)\s+(?:the\s+)?(.+?)[\.\?!]?$"
    )
    m = re.search(compound, lower)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Fall back to direct "place A on B" / "put A on B" form.
    for verb in ("place", "put", "stack", "set", "drop"):
        m = re.search(
            rf"{verb}\s+(?:up\s+)?(?:the\s+)?(.+?)\s+(?:on top of|onto|on|in|into)\s+(?:the\s+)?(.+?)[\.\?!]?$",
            lower,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def _resolve_prop(
    name: str, props: Mapping[str, tuple[float, float, float]]
) -> tuple[str, tuple[float, float, float]] | None:
    """Match an English name like 'red block' to a prop id like 'red_block'."""
    canon = re.sub(r"[\s_-]+", " ", name).strip()
    direct = canon.replace(" ", "_")
    if direct in props:
        return direct, props[direct]
    # Fuzzy match: every token must appear in the prop id (split on _)
    tokens = canon.split()
    for prop_id, pos in props.items():
        prop_tokens = prop_id.lower().split("_")
        if all(any(t in pt for pt in prop_tokens) for t in tokens):
            return prop_id, pos
    return None


def _meters_to_mm(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    return (pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0)


class PickPlaceExecutor:
    """Hand-coded pick-and-place skill.

    The executor parses the node instruction, looks up source and target
    positions in the supplied world state, and emits eight action chunks
    plus one critic note per run. The RunLoop dispatches each chunk
    through the safety gate before it reaches the driver.
    """

    def __init__(
        self,
        world_source: WorldStateSource,
        *,
        approach_height_mm: float = APPROACH_HEIGHT_MM,
        drop_height_mm: float = DROP_HEIGHT_MM,
        velocity_cap: float = DEFAULT_VELOCITY_CAP,
    ) -> None:
        self._world = world_source
        self._approach = float(approach_height_mm)
        self._drop = float(drop_height_mm)
        self._cap = float(velocity_cap)

    def execute(
        self, node: PlanNode, driver: Driver, run_id: str
    ) -> ExecResult:
        parsed = _parse_pick_place(node.instruction)
        if parsed is None:
            return ExecResult(
                ok=False,
                error=f"pick-place parser did not match instruction: {node.instruction!r}",
            )
        src_name, tgt_name = parsed
        props = dict(self._world.world_state())
        src = _resolve_prop(src_name, props)
        tgt = _resolve_prop(tgt_name, props)
        if src is None:
            return ExecResult(ok=False, error=f"unknown source prop: {src_name!r}")
        if tgt is None:
            return ExecResult(ok=False, error=f"unknown target prop: {tgt_name!r}")

        src_id, src_pos = src
        tgt_id, tgt_pos = tgt
        sx, sy, sz = _meters_to_mm(src_pos)
        tx, ty, tz = _meters_to_mm(tgt_pos)
        rpy = (math.pi, 0.0, 0.0)
        chunks: list[dict[str, Any]] = []

        def waypoint(x: float, y: float, z: float, label: str) -> None:
            chunks.append(
                {
                    "type": "action_chunk",
                    "action_space": "tcp_xyzrpy_mm",
                    "action": [x, y, z, rpy[0], rpy[1], rpy[2]],
                    "velocity_cap": self._cap,
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

        chunks.append(
            {
                "type": "critic_note",
                "text": (
                    f"executing pick-and-place: source={src_id} at "
                    f"({sx:.1f},{sy:.1f},{sz:.1f})mm → target={tgt_id} at "
                    f"({tx:.1f},{ty:.1f},{tz:.1f})mm"
                ),
            }
        )
        waypoint(sx, sy, sz + self._approach, f"above_{src_id}")
        grip("open", "open_gripper")
        waypoint(sx, sy, sz, f"at_{src_id}")
        grip("closed", f"grasp_{src_id}")
        waypoint(sx, sy, sz + self._approach, f"lift_from_{src_id}")
        waypoint(tx, ty, tz + self._approach, f"above_{tgt_id}")
        waypoint(tx, ty, tz + self._drop, f"drop_on_{tgt_id}")
        grip("open", f"release_on_{tgt_id}")
        waypoint(tx, ty, tz + self._approach, f"retract_from_{tgt_id}")

        return ExecResult(ok=True, chunks=chunks)


__all__ = [
    "HardcodedWorldState",
    "PickPlaceExecutor",
    "SkillExecutor",
    "WorldStateSource",
]
