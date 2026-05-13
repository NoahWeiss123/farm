"""Load a run record back from disk."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from farm_edge_agent.run_record.schema import Event, RunRecord

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


class RunRecordReader:
    """Iterate events from a record.jsonl, lazily."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self):
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                yield _EVENT_ADAPTER.validate_python(json.loads(line))


def load_run_record(path: Path, run_id: str | None = None) -> RunRecord:
    """Read a record.jsonl into a :class:`RunRecord`.

    If ``run_id`` is omitted, it is read from the ``run_started`` event's
    ``data.run_id`` field. If no ``run_started`` event is present, the parent
    directory name is used as a fallback.
    """
    events = list(RunRecordReader(path))
    if run_id is None:
        run_id = _extract_run_id(events) or path.parent.name
    return RunRecord(run_id=run_id, events=events)


def _extract_run_id(events: list[Event]) -> str | None:
    for event in events:
        if event.type == "run_started":
            return event.data.run_id
    return None
