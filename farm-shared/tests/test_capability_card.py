import copy

import pytest
from farm_shared.capability_card import (
    ACTION_SPACES,
    CapabilityCard,
    CapabilityCardError,
    load_schema,
)

VALID_CARD: dict = {
    "id": "pi05-ufactory-ft-v1",
    "name": "π0.5 (fine-tuned on UFactory 850)",
    "roles": ["controller"],
    "embodiment": {
        "arm": "ufactory-850",
        "dof": 6,
        "action_space": "ee_pose_delta_base_frame",
        "control_rate_hz": 30,
    },
    "input_modalities": ["rgb_image", "joint_state", "language"],
    "camera_views": ["wrist", "overhead_optional"],
    "skills": [
        {"pick": {"confidence": 0.8, "learned_from": "240_demos"}},
        {"place": {"confidence": 0.85, "learned_from": "240_demos"}},
        {"stack": {"confidence": 0.7, "learned_from": "60_demos"}},
        {"pour": {"confidence": 0.4, "learned_from": "0_demos"}},
    ],
    "latency": {"p50_ms_per_chunk": 95, "p99_ms_per_chunk": 220},
    "cost_per_chunk_usd": 0.0008,
    "determinism": "stochastic",
    "safety": {"requires_envelope": True, "supports_velocity_cap": True},
    "fallbacks": ["classical-planner-pick", "gemini-robotics-act"],
}


def test_design_md_example_loads_round_trip():
    card = CapabilityCard.from_dict(VALID_CARD)
    assert card.id == "pi05-ufactory-ft-v1"
    assert card.embodiment.arm == "ufactory-850"
    assert card.embodiment.dof == 6
    assert card.embodiment.action_space == "ee_pose_delta_base_frame"
    assert card.determinism == "stochastic"
    assert card.fallbacks == ["classical-planner-pick", "gemini-robotics-act"]
    assert len(card.skills) == 4


def test_unknown_action_space_emits_did_you_mean():
    bad = copy.deepcopy(VALID_CARD)
    bad["embodiment"]["action_space"] = "ee_pose_delta_basefrm"
    with pytest.raises(CapabilityCardError) as exc:
        CapabilityCard.from_dict(bad)
    msg = str(exc.value)
    assert "ee_pose_delta_basefrm" in msg
    assert "Did you mean 'ee_pose_delta_base_frame'?" in msg
    assert "https://farm.dev/schemas/capability_card.v1" in msg


def test_unknown_determinism_emits_did_you_mean():
    bad = copy.deepcopy(VALID_CARD)
    bad["determinism"] = "stochastik"
    with pytest.raises(CapabilityCardError) as exc:
        CapabilityCard.from_dict(bad)
    assert "stochastik" in str(exc.value)
    assert "Did you mean 'stochastic'?" in str(exc.value)


def test_missing_required_field_rejected():
    bad = copy.deepcopy(VALID_CARD)
    del bad["embodiment"]["action_space"]
    with pytest.raises(CapabilityCardError):
        CapabilityCard.from_dict(bad)


def test_schema_lists_all_action_spaces():
    schema = load_schema()
    enum = schema["properties"]["embodiment"]["properties"]["action_space"]["enum"]
    assert tuple(enum) == ACTION_SPACES


def test_role_must_be_known():
    bad = copy.deepcopy(VALID_CARD)
    bad["roles"] = ["plannar"]
    with pytest.raises(CapabilityCardError):
        CapabilityCard.from_dict(bad)
