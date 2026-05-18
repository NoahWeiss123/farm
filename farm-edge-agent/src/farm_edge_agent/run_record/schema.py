"""Pydantic types for run-record events.

A run record is an ordered JSONL stream of events. Every event has the same outer
shape: ``ts`` (unix seconds, float), ``type`` (event-kind tag), ``data`` (payload).
The payload schema is selected by ``type`` — see :data:`EVENT_TYPES`.

Each event must fit in 4 KiB so the log stays cheap to append and tail; large blobs
(camera frames, model thinking traces) are written to sidecar files and referenced
here by relative path.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_EVENT_BYTES = 4096


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ts: float
    type: str


class RunStartedData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task: str
    workspace: str
    agent_version: str
    protocol_version: str
    calibration_hash: str
    config_snapshot: dict[str, Any]


class RunStarted(_EventBase):
    type: Literal["run_started"] = "run_started"
    data: RunStartedData


class PlanEmittedData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    nodes: list[dict[str, Any]]
    router_reason: str | None = None


class PlanEmitted(_EventBase):
    type: Literal["plan_emitted"] = "plan_emitted"
    data: PlanEmittedData


class NodeStartedData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    backend: str


class NodeStarted(_EventBase):
    type: Literal["node_started"] = "node_started"
    data: NodeStartedData


class ActionChunkData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    chunk_index: int
    step_index: int
    action: list[float]
    action_space: str
    # Human-readable label for the chunk (e.g. "above_red_block",
    # "grasp_red_block"). Used by the dashboard's "what's it doing now"
    # surface. Optional so older records (pre-skill-labels) still parse.
    label: str | None = None


class ActionChunk(_EventBase):
    type: Literal["action_chunk"] = "action_chunk"
    data: ActionChunkData


class ObsChunkData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    chunk_index: int
    step_index: int
    joint_state: list[float]
    ee_pose: list[float]
    image_paths: dict[str, str] = Field(default_factory=dict)


class ObsChunk(_EventBase):
    type: Literal["obs_chunk"] = "obs_chunk"
    data: ObsChunkData


class SafetyEventData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str | None
    kind: str
    detail: str


class SafetyEvent(_EventBase):
    type: Literal["safety_event"] = "safety_event"
    data: SafetyEventData


class FallbackInvokedData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    from_backend: str
    to_backend: str
    trigger: str


class FallbackInvoked(_EventBase):
    type: Literal["fallback_invoked"] = "fallback_invoked"
    data: FallbackInvokedData


class RecoveryInvokedData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    primitive: str


class RecoveryInvoked(_EventBase):
    type: Literal["recovery_invoked"] = "recovery_invoked"
    data: RecoveryInvokedData


class NodeCompletedData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    outcome: str


class NodeCompleted(_EventBase):
    type: Literal["node_completed"] = "node_completed"
    data: NodeCompletedData


class RunCompletedData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    outcome: str
    wall_clock_s: float


class RunCompleted(_EventBase):
    type: Literal["run_completed"] = "run_completed"
    data: RunCompletedData


class CriticNoteData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str | None
    text: str


class CriticNote(_EventBase):
    type: Literal["critic_note"] = "critic_note"
    data: CriticNoteData


Event = Annotated[
    RunStarted
    | PlanEmitted
    | NodeStarted
    | ActionChunk
    | ObsChunk
    | SafetyEvent
    | FallbackInvoked
    | RecoveryInvoked
    | NodeCompleted
    | RunCompleted
    | CriticNote,
    Field(discriminator="type"),
]


EVENT_TYPES: dict[str, type[_EventBase]] = {
    "run_started": RunStarted,
    "plan_emitted": PlanEmitted,
    "node_started": NodeStarted,
    "action_chunk": ActionChunk,
    "obs_chunk": ObsChunk,
    "safety_event": SafetyEvent,
    "fallback_invoked": FallbackInvoked,
    "recovery_invoked": RecoveryInvoked,
    "node_completed": NodeCompleted,
    "run_completed": RunCompleted,
    "critic_note": CriticNote,
}


class RunRecord(BaseModel):
    """An ordered list of events read back from a record.jsonl."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    events: list[Event]

    @property
    def started(self) -> RunStarted | None:
        for event in self.events:
            if isinstance(event, RunStarted):
                return event
        return None
