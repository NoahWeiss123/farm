from farm_edge_agent.run_record.exporter import to_lerobot_shards
from farm_edge_agent.run_record.reader import RunRecordReader, load_run_record
from farm_edge_agent.run_record.schema import (
    EVENT_TYPES,
    MAX_EVENT_BYTES,
    ActionChunk,
    CriticNote,
    Event,
    FallbackInvoked,
    NodeCompleted,
    NodeStarted,
    ObsChunk,
    PlanEmitted,
    RecoveryInvoked,
    RunCompleted,
    RunRecord,
    RunStarted,
    SafetyEvent,
)
from farm_edge_agent.run_record.writer import RunRecordWriter


class FarmError(Exception):
    """Run-record-scoped FARM error.

    Carries a stable code in the ``FARM-Exxxx`` namespace plus a one-line message.
    A full catalog lives in ``farm_edge_agent.errors`` once that module lands;
    this minimal shim exists so ``farm export`` can raise on missing run-ids.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


__all__ = [
    "EVENT_TYPES",
    "MAX_EVENT_BYTES",
    "ActionChunk",
    "CriticNote",
    "Event",
    "FallbackInvoked",
    "FarmError",
    "NodeCompleted",
    "NodeStarted",
    "ObsChunk",
    "PlanEmitted",
    "RecoveryInvoked",
    "RunCompleted",
    "RunRecord",
    "RunRecordReader",
    "RunRecordWriter",
    "RunStarted",
    "SafetyEvent",
    "load_run_record",
    "to_lerobot_shards",
]
