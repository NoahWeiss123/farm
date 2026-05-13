from __future__ import annotations

import base64
from typing import Annotated, Any, Literal

from farm_shared.errors import ErrorCode
from pydantic import BaseModel, Field, field_serializer, field_validator

GripperState = Literal["open", "closed", "grasping"]
GripperCommand = Literal["open", "close", "hold"]
ControlCommand = Literal["pause", "resume", "abort", "home"]


class TcpPose(BaseModel):
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class EePoseDelta(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0
    gripper: GripperCommand | None = None


class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    protocol_version: str
    agent_version: str
    feature_flags: dict[str, bool] = Field(default_factory=dict)


class Ack(BaseModel):
    type: Literal["ack"] = "ack"
    protocol_version: str
    accepted: bool
    reason: str | None = None


class ObsChunk(BaseModel):
    type: Literal["obs_chunk"] = "obs_chunk"
    run_id: str
    ts: float
    frames: dict[str, bytes | str]
    joint_state: list[float]
    tcp_pose: TcpPose
    gripper_state: GripperState

    @field_serializer("frames")
    def _serialize_frames(self, frames: dict[str, bytes | str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in frames.items():
            if isinstance(value, bytes):
                out[name] = base64.b64encode(value).decode("ascii")
            else:
                out[name] = value
        return out


class ActionChunk(BaseModel):
    type: Literal["action_chunk"] = "action_chunk"
    run_id: str
    chunk_id: int
    actions: list[EePoseDelta]
    suggested_dwell_ms: int


class SafetyEvent(BaseModel):
    type: Literal["safety_event"] = "safety_event"
    run_id: str
    ts: float
    code: ErrorCode
    halted: bool

    @field_validator("code", mode="before")
    @classmethod
    def _coerce_code(cls, v: Any) -> ErrorCode:
        if isinstance(v, ErrorCode):
            return v
        if isinstance(v, str):
            return ErrorCode[v]
        raise ValueError(f"unrecognized error code: {v!r}")

    @field_serializer("code")
    def _serialize_code(self, code: ErrorCode) -> str:
        return code.name


class Control(BaseModel):
    type: Literal["control"] = "control"
    run_id: str
    command: ControlCommand


Message = Annotated[
    Hello | Ack | ObsChunk | ActionChunk | SafetyEvent | Control,
    Field(discriminator="type"),
]


_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "hello": Hello,
    "ack": Ack,
    "obs_chunk": ObsChunk,
    "action_chunk": ActionChunk,
    "safety_event": SafetyEvent,
    "control": Control,
}


def parse_message(raw: str | bytes) -> BaseModel:
    """Decode a JSON-encoded wire message into the matching pydantic model.

    Tag-dispatched on the ``type`` field; raises ``ValueError`` on unknown types.
    """
    import json

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("wire message must be a JSON object")
    mtype = data.get("type")
    model = _TYPE_TO_MODEL.get(mtype)
    if model is None:
        raise ValueError(f"unknown message type: {mtype!r}")
    return model.model_validate(data)
