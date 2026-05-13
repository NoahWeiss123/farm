"""Append-only JSONL writer for a single run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from farm_edge_agent.run_record.schema import EVENT_TYPES, MAX_EVENT_BYTES

DEFAULT_RUNS_ROOT = Path.home() / ".farm" / "runs"


def _runs_root() -> Path:
    override = os.environ.get("FARM_RUNS_DIR")
    return Path(override) if override else DEFAULT_RUNS_ROOT


class RunRecordWriter:
    """Append events for a single run to ``<root>/<run-id>/record.jsonl``.

    Each ``write`` serializes the event with sorted keys, asserts the encoded
    byte length is ≤ ``MAX_EVENT_BYTES``, and flushes so a crash mid-run still
    leaves a readable prefix on disk.
    """

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        base = (root if root is not None else _runs_root()) / run_id
        base.mkdir(parents=True, exist_ok=True)
        self.run_dir = base
        self.path = base / "record.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: BaseModel | dict[str, Any]) -> None:
        payload = self._as_dict(event)
        self._validate(payload)
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        encoded = line.encode("utf-8")
        if len(encoded) > MAX_EVENT_BYTES:
            raise ValueError(
                f"event of type {payload.get('type')!r} is {len(encoded)} bytes "
                f"(> {MAX_EVENT_BYTES}); write large blobs as sidecars and reference by path"
            )
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> RunRecordWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _as_dict(event: BaseModel | dict[str, Any]) -> dict[str, Any]:
        if isinstance(event, BaseModel):
            return event.model_dump(mode="json")
        return dict(event)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {kind!r}")
        EVENT_TYPES[kind].model_validate(payload)
