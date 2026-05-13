"""Tests for the classical controller and its capability card."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from farm_edge_agent.backends import (
    ActionChunk,
    ClassicalController,
    ClassicalSkillNotFoundError,
    Observation,
    Pose,
    Waypoint,
)
from farm_edge_agent.backends.classical import (
    DEFAULT_APPROACH_HEIGHT_MM,
    DEFAULT_CARD_PATH,
    DEFAULT_STACK_CLEARANCE_MM,
    SKILL_PICK,
    SKILL_PLACE,
    SKILL_STACK,
    PerceptionCallable,
)
from farm_edge_agent.capability_cards.loader import load_card
from farm_edge_agent.errors import FarmError

RED_BLOCK: Pose = (300.0, 50.0, 100.0, 180.0, 0.0, 0.0)
BLUE_BLOCK: Pose = (350.0, -50.0, 100.0, 180.0, 0.0, 0.0)
DROP_ZONE: Pose = (400.0, 0.0, 120.0, 180.0, 0.0, 0.0)

START_OBS = Observation(
    joint_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    tcp_pose=(200.0, 0.0, 300.0, 180.0, 0.0, 0.0),
    gripper="open",
)


async def _yield(*items: Observation) -> AsyncIterator[Observation]:
    for item in items:
        yield item


def _perception(scene: dict[str, Pose]) -> PerceptionCallable:
    def _stub(_obs: Observation, _instruction: str) -> dict[str, Pose]:
        return dict(scene)

    return _stub


def _above(pose: Pose, lift: float) -> Pose:
    return (pose[0], pose[1], pose[2] + lift, pose[3], pose[4], pose[5])


async def _collect(controller: ClassicalController, instruction: str) -> list[ActionChunk]:
    chunks: list[ActionChunk] = []
    async for chunk in controller.act(_yield(START_OBS), instruction):
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_pick_emits_expected_waypoint_sequence() -> None:
    controller = ClassicalController(perception=_perception({"target": RED_BLOCK}))
    chunks = await _collect(controller, "pick the red block")
    above = _above(RED_BLOCK, DEFAULT_APPROACH_HEIGHT_MM)
    assert chunks == [
        ActionChunk(
            skill=SKILL_PICK,
            step_index=0,
            waypoints=(Waypoint(pose=above, gripper="open", label="approach"),),
        ),
        ActionChunk(
            skill=SKILL_PICK,
            step_index=1,
            waypoints=(Waypoint(pose=RED_BLOCK, label="descend"),),
        ),
        ActionChunk(
            skill=SKILL_PICK,
            step_index=2,
            waypoints=(Waypoint(pose=RED_BLOCK, gripper="closed", label="grasp"),),
        ),
        ActionChunk(
            skill=SKILL_PICK,
            step_index=3,
            waypoints=(Waypoint(pose=above, label="lift"),),
            terminal=True,
        ),
    ]


@pytest.mark.asyncio
async def test_place_emits_expected_waypoint_sequence() -> None:
    controller = ClassicalController(perception=_perception({"destination": DROP_ZONE}))
    chunks = await _collect(controller, "place at the drop zone")
    above = _above(DROP_ZONE, DEFAULT_APPROACH_HEIGHT_MM)
    assert chunks == [
        ActionChunk(
            skill=SKILL_PLACE,
            step_index=0,
            waypoints=(Waypoint(pose=above, label="approach"),),
        ),
        ActionChunk(
            skill=SKILL_PLACE,
            step_index=1,
            waypoints=(Waypoint(pose=DROP_ZONE, label="descend"),),
        ),
        ActionChunk(
            skill=SKILL_PLACE,
            step_index=2,
            waypoints=(Waypoint(pose=DROP_ZONE, gripper="open", label="release"),),
        ),
        ActionChunk(
            skill=SKILL_PLACE,
            step_index=3,
            waypoints=(Waypoint(pose=above, label="lift"),),
            terminal=True,
        ),
    ]


@pytest.mark.asyncio
async def test_stack_concatenates_pick_then_place_above_destination() -> None:
    controller = ClassicalController(
        perception=_perception({"source": RED_BLOCK, "destination": BLUE_BLOCK}),
    )
    chunks = await _collect(controller, "stack the red block on the blue block")

    pick_above = _above(RED_BLOCK, DEFAULT_APPROACH_HEIGHT_MM)
    release = _above(BLUE_BLOCK, DEFAULT_STACK_CLEARANCE_MM)
    place_above = _above(release, DEFAULT_APPROACH_HEIGHT_MM)

    assert [c.step_index for c in chunks] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert all(c.skill == SKILL_STACK for c in chunks)
    assert all(not c.terminal for c in chunks[:-1])
    assert chunks[-1].terminal is True

    assert chunks[0].waypoints == (Waypoint(pose=pick_above, gripper="open", label="approach"),)
    assert chunks[1].waypoints == (Waypoint(pose=RED_BLOCK, label="descend"),)
    assert chunks[2].waypoints == (
        Waypoint(pose=RED_BLOCK, gripper="closed", label="grasp"),
    )
    assert chunks[3].waypoints == (Waypoint(pose=pick_above, label="lift"),)

    assert chunks[4].waypoints == (Waypoint(pose=place_above, label="approach"),)
    assert chunks[5].waypoints == (Waypoint(pose=release, label="descend"),)
    assert chunks[6].waypoints == (Waypoint(pose=release, gripper="open", label="release"),)
    assert chunks[7].waypoints == (Waypoint(pose=place_above, label="lift"),)


@pytest.mark.asyncio
async def test_act_is_deterministic_across_repeated_runs() -> None:
    scene = {"source": RED_BLOCK, "destination": BLUE_BLOCK}
    controller = ClassicalController(perception=_perception(scene))
    first = await _collect(controller, "stack red on blue")
    second = await _collect(controller, "stack red on blue")
    assert first == second


@pytest.mark.asyncio
async def test_unknown_skill_raises_farm_error() -> None:
    controller = ClassicalController(perception=_perception({"target": RED_BLOCK}))

    with pytest.raises(ClassicalSkillNotFoundError) as exc_info:
        await _collect(controller, "fly the arm to the moon")

    assert isinstance(exc_info.value, FarmError)
    rendered = str(exc_info.value)
    assert rendered.startswith("[FARM-E2002]")
    assert "fly the arm to the moon" in rendered
    assert "pick" in rendered
    assert "place" in rendered
    assert "stack" in rendered


def test_capability_card_validates_against_schema() -> None:
    card = load_card(DEFAULT_CARD_PATH)
    assert card.id == "classical-planner-v1"
    assert card.roles == ["controller"]
    assert card.determinism == "deterministic"
    assert card.embodiment.arm == "ufactory-850"
    assert card.embodiment.action_space == "ee_pose_delta_base_frame"

    skill_names = [next(iter(s)) for s in card.skills]
    assert skill_names == ["pick", "place", "stack"]
    confidences = {next(iter(s)): next(iter(s.values()))["confidence"] for s in card.skills}
    assert confidences == {"pick": 1.0, "place": 1.0, "stack": 0.9}


def test_capability_card_lives_next_to_module() -> None:
    """The card must ship inside the package so it travels with the wheel."""
    assert DEFAULT_CARD_PATH.exists()
    assert DEFAULT_CARD_PATH.parent.name == "capability_cards"
    assert DEFAULT_CARD_PATH.parent.parent.name == "backends"
