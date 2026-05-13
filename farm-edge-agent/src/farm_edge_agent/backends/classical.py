"""Deterministic classical controller for the Edge Agent.

Implements the ``pick``, ``place``, and ``stack`` skills as fixed waypoint state
machines parameterised by object poses from a caller-supplied ``perception``
callable. The xArm SDK runs the IK at execution time; this module only emits
absolute base-frame TCP waypoints and gripper transitions.

The contract is deterministic: same observation + same instruction → identical
``ActionChunk`` stream. This is the architectural commitment that makes the
classical leg of the fallback chain auditable (DESIGN.md "Fallback chain").
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from pathlib import Path

from farm_shared.capability_card import CapabilityCard
from farm_shared.errors import ErrorCode

from farm_edge_agent.capability_cards.loader import load_card
from farm_edge_agent.errors import FarmError

from .base import (
    ActionChunk,
    GripperState,
    Observation,
    Pose,
    Waypoint,
)

DEFAULT_CARD_PATH = Path(__file__).parent / "capability_cards" / "classical.yaml"

DEFAULT_APPROACH_HEIGHT_MM = 80.0
"""Vertical offset above the target used for approach and lift waypoints.

80 mm clears the default block stack height (~50 mm) with margin while staying
well inside the conservative envelope from DESIGN.md "Safety"."""

DEFAULT_STACK_CLEARANCE_MM = 50.0
"""Vertical offset added to the destination pose when releasing a stacked block.

Matches the nominal coloured-block height from the primary eval task; one cube
sits on top of another with a small air gap to absorb pose noise."""

SKILL_PICK = "pick"
SKILL_PLACE = "place"
SKILL_STACK = "stack"

SCENE_KEY_TARGET = "target"
SCENE_KEY_DESTINATION = "destination"
SCENE_KEY_SOURCE = "source"

PerceptionCallable = Callable[[Observation, str], dict[str, Pose]]
"""Caller-supplied stub for object pose detection.

Real perception is its own subsystem; the classical controller never touches a
camera frame directly. The callable receives the inbound observation and the
operator's instruction, and returns a dict of well-known scene keys
(``target``, ``destination``, ``source``) → 6-DoF base-frame TCP poses.
"""


class ClassicalSkillNotFoundError(FarmError):
    """Raised when an instruction does not name a registered classical skill.

    Subclasses :class:`farm_edge_agent.errors.FarmError` so callers can
    ``except FarmError``. The shared catalog does not yet expose a dedicated
    slot for this case; we render the canonical
    ``[FARM-Exxxx] ... — fix: ...`` shape with a placeholder code until the
    catalog gains its own entry (see ``tasks/_followups.md``).
    """

    _PLACEHOLDER_CODE = "FARM-E2002"

    def __init__(self, skill: str, registry: Iterable[str]) -> None:
        self.skill = skill
        self.registry = tuple(sorted(registry))
        available = ", ".join(self.registry) if self.registry else "(empty)"
        self.code = ErrorCode.E2001
        # ErrorCode.E2001's template needs `value` and `suggestion`; populate
        # both so `format_error(code, **slots)` still renders if a caller
        # routes us through the shared formatter.
        self.slots = {"value": skill, "suggestion": available}
        message = (
            f"[{self._PLACEHOLDER_CODE}] classical controller has no skill "
            f"'{skill}' — fix: pick one of: {available}"
        )
        Exception.__init__(self, message)

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def _above(pose: Pose, lift_mm: float) -> Pose:
    return (pose[0], pose[1], pose[2] + lift_mm, pose[3], pose[4], pose[5])


class ClassicalController:
    """Deterministic ``pick`` / ``place`` / ``stack`` controller.

    Reads the classical capability card from disk on construction so the card
    and the implementation cannot drift; tests that need a different card may
    pass one explicitly via ``capability_card``.

    The skill registry is closed by design — the classical leg is the
    reliability floor (DESIGN.md "Fallback chain") and new skills cost a card
    update plus a tested state machine.
    """

    def __init__(
        self,
        *,
        perception: PerceptionCallable,
        capability_card: CapabilityCard | None = None,
        approach_height_mm: float = DEFAULT_APPROACH_HEIGHT_MM,
        stack_clearance_mm: float = DEFAULT_STACK_CLEARANCE_MM,
        open_state: GripperState = "open",
        grasp_state: GripperState = "closed",
    ) -> None:
        if capability_card is None:
            capability_card = load_card(DEFAULT_CARD_PATH)
        self.capability_card = capability_card
        self._perception = perception
        self._approach_height = approach_height_mm
        self._stack_clearance = stack_clearance_mm
        self._open = open_state
        self._grasp = grasp_state
        self._skills: dict[str, Callable[[dict[str, Pose]], Iterator[ActionChunk]]] = {
            SKILL_PICK: self._pick_chunks,
            SKILL_PLACE: self._place_chunks,
            SKILL_STACK: self._stack_chunks,
        }

    @property
    def skills(self) -> tuple[str, ...]:
        return tuple(self._skills)

    async def act(
        self,
        obs_stream: AsyncIterator[Observation],
        instruction: str,
    ) -> AsyncIterator[ActionChunk]:
        """Emit a deterministic chunk sequence for the skill named in ``instruction``."""

        obs = await _first_observation(obs_stream)
        skill = self._parse_skill(instruction)
        scene = self._perception(obs, instruction)
        for chunk in self._skills[skill](scene):
            yield chunk

    def _parse_skill(self, instruction: str) -> str:
        lowered = instruction.lower()
        for skill in self._skills:
            if skill in lowered:
                return skill
        raise ClassicalSkillNotFoundError(instruction, self._skills.keys())

    def _pick_chunks(self, scene: dict[str, Pose]) -> Iterator[ActionChunk]:
        target = _scene_pose(scene, SCENE_KEY_TARGET)
        yield from self._pick_waypoints(SKILL_PICK, 0, target)

    def _place_chunks(self, scene: dict[str, Pose]) -> Iterator[ActionChunk]:
        destination = _scene_pose(scene, SCENE_KEY_DESTINATION)
        yield from self._place_waypoints(SKILL_PLACE, 0, destination, lift_clearance=0.0)

    def _stack_chunks(self, scene: dict[str, Pose]) -> Iterator[ActionChunk]:
        source = _scene_pose(scene, SCENE_KEY_SOURCE)
        destination = _scene_pose(scene, SCENE_KEY_DESTINATION)
        index = 0
        for chunk in self._pick_waypoints(SKILL_STACK, index, source, terminal=False):
            yield chunk
            index = chunk.step_index + 1
        yield from self._place_waypoints(
            SKILL_STACK,
            index,
            destination,
            lift_clearance=self._stack_clearance,
        )

    def _pick_waypoints(
        self,
        skill: str,
        start_index: int,
        target: Pose,
        *,
        terminal: bool = True,
    ) -> Iterator[ActionChunk]:
        above = _above(target, self._approach_height)
        yield ActionChunk(
            skill=skill,
            step_index=start_index,
            waypoints=(Waypoint(pose=above, gripper=self._open, label="approach"),),
        )
        yield ActionChunk(
            skill=skill,
            step_index=start_index + 1,
            waypoints=(Waypoint(pose=target, label="descend"),),
        )
        yield ActionChunk(
            skill=skill,
            step_index=start_index + 2,
            waypoints=(Waypoint(pose=target, gripper=self._grasp, label="grasp"),),
        )
        yield ActionChunk(
            skill=skill,
            step_index=start_index + 3,
            waypoints=(Waypoint(pose=above, label="lift"),),
            terminal=terminal,
        )

    def _place_waypoints(
        self,
        skill: str,
        start_index: int,
        destination: Pose,
        *,
        lift_clearance: float,
    ) -> Iterator[ActionChunk]:
        release_pose = _above(destination, lift_clearance)
        above = _above(release_pose, self._approach_height)
        yield ActionChunk(
            skill=skill,
            step_index=start_index,
            waypoints=(Waypoint(pose=above, label="approach"),),
        )
        yield ActionChunk(
            skill=skill,
            step_index=start_index + 1,
            waypoints=(Waypoint(pose=release_pose, label="descend"),),
        )
        yield ActionChunk(
            skill=skill,
            step_index=start_index + 2,
            waypoints=(Waypoint(pose=release_pose, gripper=self._open, label="release"),),
        )
        yield ActionChunk(
            skill=skill,
            step_index=start_index + 3,
            waypoints=(Waypoint(pose=above, label="lift"),),
            terminal=True,
        )


def _scene_pose(scene: dict[str, Pose], key: str) -> Pose:
    pose = scene[key]
    if len(pose) != 6:
        raise ValueError(
            f"perception returned pose with len={len(pose)} for '{key}'; expected 6"
        )
    return (
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
    )


async def _first_observation(stream: AsyncIterator[Observation]) -> Observation:
    async for obs in stream:
        return obs
    raise ValueError("obs_stream produced no observations before completing")
