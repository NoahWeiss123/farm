"""Public Python API for FARM.

`from farm import Client` is the entry point that research labs use to drive
FARM from notebooks. The CLI is a thin wrapper over this same surface.

The Client talks to the dispatcher through a `Transport` interface. Tests
inject a fake transport; the production transport (WebSocket to the cloud
dispatcher) is wired in by later tasks. `runs.list` and `runs.export` read
the local run-record directory from task 010.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class FarmError(Exception):
    """Raised by the Python API for user-facing failures."""


@dataclass(frozen=True)
class Event:
    """One event from a run. Matches the run-record event shape from task 010."""

    type: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        return cls(type=d["type"], ts=float(d.get("ts", 0.0)), data=dict(d.get("data", {})))


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    task: str
    task_id: str | None
    backend: str
    status: str
    started_at: float | None
    completed_at: float | None
    workspace: str | None


@dataclass(frozen=True)
class CapabilityCard:
    id: str
    skills: list[str]
    raw: dict[str, Any] = field(default_factory=dict)


class Transport(Protocol):
    """How a Client reaches the dispatcher. Real impl uses a WebSocket."""

    def start_run(
        self,
        *,
        task: str,
        task_id: str | None,
        backend: str,
        workspace: str | None,
        api_key: str | None,
    ) -> dict[str, Any]: ...

    def stream_events(self, run_id: str) -> Iterator[Event]: ...


class Run:
    """Handle on a single dispatched run.

    Events are streamed from the transport once and cached, so callers can
    replay them via `events()` after `wait()` has already drained the stream.
    """

    def __init__(
        self,
        run_id: str,
        task: str,
        task_id: str | None,
        backend: str,
        transport: Transport,
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.task_id = task_id
        self.backend = backend
        self._transport = transport
        self._events: list[Event] = []
        self._stream: Iterator[Event] | None = None
        self._drained = False

    def _stream_iter(self) -> Iterator[Event]:
        if self._stream is None:
            self._stream = iter(self._transport.stream_events(self.run_id))
        return self._stream

    async def events(self) -> AsyncIterator[Event]:
        if self._drained:
            for ev in self._events:
                yield ev
            return
        stream = self._stream_iter()
        for ev in stream:
            self._events.append(ev)
            yield ev
            await asyncio.sleep(0)
        self._drained = True

    def wait(self) -> RunSummary:
        if not self._drained:
            for ev in self._stream_iter():
                self._events.append(ev)
            self._drained = True
        return self._build_summary()

    def _build_summary(self) -> RunSummary:
        status = "running"
        started_at: float | None = None
        completed_at: float | None = None
        workspace: str | None = None
        for ev in self._events:
            if ev.type == "run_started":
                started_at = ev.ts
                workspace = ev.data.get("workspace")
            elif ev.type == "run_completed":
                status = str(ev.data.get("status", "completed"))
                completed_at = ev.ts
        return RunSummary(
            run_id=self.run_id,
            task=self.task,
            task_id=self.task_id,
            backend=self.backend,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            workspace=workspace,
        )

    def to_dataframe(self) -> Any:
        """Return cached events shaped like LeRobot frames.

        Pandas is a soft dep: importing it lazily keeps the wheel small and
        lets users without pandas still use the rest of the Client.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError as e:
            raise FarmError(
                "pandas is required for to_dataframe(). "
                "fix: 'pip install pandas' or 'pip install farm-edge-agent[pandas]'"
            ) from e
        rows: list[dict[str, Any]] = []
        for ev in self._events:
            if ev.type != "action_chunk":
                continue
            row: dict[str, Any] = {"ts": ev.ts, "action": ev.data.get("action")}
            obs = ev.data.get("obs") or {}
            if isinstance(obs, dict):
                for k, v in obs.items():
                    row[f"observation.{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)


class _RunsNamespace:
    """`client.runs.list(...)`, `client.runs.export(...)`."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    def list(
        self, workspace: str | None = None, since: str | datetime | None = None
    ) -> list[RunSummary]:
        if not self._runs_dir.exists():
            return []
        since_ts = _parse_since(since)
        out: list[RunSummary] = []
        for d in sorted(self._runs_dir.iterdir()):
            if not d.is_dir():
                continue
            record = d / "record.jsonl"
            if not record.exists():
                continue
            summary = _summary_from_record(d.name, record)
            if workspace is not None and summary.workspace != workspace:
                continue
            if since_ts is not None and (summary.started_at or 0.0) < since_ts:
                continue
            out.append(summary)
        return out

    def export(self, run_id: str, format: str = "jsonl") -> Path:
        run_dir = self._runs_dir / run_id
        record = run_dir / "record.jsonl"
        if not record.exists():
            raise FarmError(f"run not found: {run_id} (no record at {record})")
        if format == "jsonl":
            out = run_dir / "export.jsonl"
            out.write_bytes(record.read_bytes())
            return out
        if format == "lerobot":
            raise FarmError(
                "lerobot export lives in farm_edge_agent.run_record.exporter; "
                "use format='jsonl' here or call `farm export <run-id> --format lerobot`"
            )
        raise FarmError(f"unknown export format: {format!r} (try 'jsonl' or 'lerobot')")


class _CardsNamespace:
    """`client.cards.list()`."""

    def __init__(self, cards_dir: Path) -> None:
        self._cards_dir = cards_dir

    def list(self) -> list[CapabilityCard]:
        if not self._cards_dir.exists():
            return []
        out: list[CapabilityCard] = []
        for p in sorted(self._cards_dir.iterdir()):
            if not p.is_file():
                continue
            raw = _load_card_file(p)
            if raw is None:
                continue
            out.append(
                CapabilityCard(
                    id=str(raw.get("id") or p.stem),
                    skills=list(raw.get("skills") or []),
                    raw=raw,
                )
            )
        return out


class Client:
    """`from farm import Client`.

    Args mirror the YAML config file. When omitted, values are read from
    `~/.farm/config.yaml` (or `$FARM_CONFIG`). The `transport` kwarg is for
    tests and advanced users; the default raises a clear error until the
    real WebSocket transport is wired up.
    """

    def __init__(
        self,
        api_key: str | None = None,
        workspace: str | None = None,
        config_path: str | os.PathLike[str] | None = None,
        *,
        transport: Transport | None = None,
        runs_dir: Path | None = None,
        cards_dir: Path | None = None,
    ) -> None:
        cfg = _load_config(config_path)
        self.api_key = api_key or cfg.get("api_key") or os.environ.get("FARM_API_KEY")
        self.workspace = workspace or cfg.get("workspace")
        self._transport = transport
        farm_home = Path(os.environ.get("FARM_HOME", str(Path.home() / ".farm")))
        self.runs_dir = runs_dir or farm_home / "runs"
        self.cards_dir = cards_dir or farm_home / "cards"
        self.runs = _RunsNamespace(self.runs_dir)
        self.cards = _CardsNamespace(self.cards_dir)

    def run(
        self,
        task: str,
        task_id: str | None = None,
        backend: str = "auto",
    ) -> Run:
        if self._transport is None:
            raise FarmError(
                "no transport configured. "
                "fix: pass transport=... (the production WebSocket transport "
                "is wired up by the dispatcher tasks)"
            )
        record = self._transport.start_run(
            task=task,
            task_id=task_id,
            backend=backend,
            workspace=self.workspace,
            api_key=self.api_key,
        )
        run_id = record.get("run_id")
        if not run_id:
            raise FarmError("transport.start_run did not return a run_id")
        return Run(
            run_id=str(run_id),
            task=task,
            task_id=task_id,
            backend=backend,
            transport=self._transport,
        )


def _load_config(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if path is None:
        env = os.environ.get("FARM_CONFIG")
        if env:
            path = env
        else:
            default = Path.home() / ".farm" / "config.yaml"
            if not default.exists():
                return {}
            path = default
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise FarmError(
            f"reading {p} requires PyYAML. "
            "fix: 'pip install PyYAML' or pass api_key/workspace explicitly"
        ) from e
    loaded = yaml.safe_load(p.read_text()) or {}
    if not isinstance(loaded, dict):
        raise FarmError(f"config file {p} did not parse to a mapping")
    return _expand_env(loaded)


def _expand_env(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            out[k] = os.environ.get(v[2:-1], v)
        elif isinstance(v, dict):
            out[k] = _expand_env(v)
        else:
            out[k] = v
    return out


def _summary_from_record(run_id: str, record_path: Path) -> RunSummary:
    task = ""
    task_id: str | None = None
    backend = "auto"
    status = "running"
    started_at: float | None = None
    completed_at: float | None = None
    workspace: str | None = None
    with record_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                raise FarmError(f"malformed run record {record_path}: {e}") from e
            ev_type = ev.get("type")
            ts = float(ev.get("ts", 0.0))
            data = ev.get("data") or {}
            if ev_type == "run_started":
                started_at = ts
                task = str(data.get("task", task))
                task_id = data.get("task_id", task_id)
                backend = str(data.get("backend", backend))
                workspace = data.get("workspace", workspace)
            elif ev_type == "run_completed":
                completed_at = ts
                status = str(data.get("status", "completed"))
    return RunSummary(
        run_id=run_id,
        task=task,
        task_id=task_id,
        backend=backend,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        workspace=workspace,
    )


def _parse_since(since: str | datetime | None) -> float | None:
    if since is None:
        return None
    if isinstance(since, datetime):
        dt = since
    else:
        dt = datetime.fromisoformat(since)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _load_card_file(p: Path) -> dict[str, Any] | None:
    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            loaded = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise FarmError(f"malformed capability card {p}: {e}") from e
        return loaded if isinstance(loaded, dict) else None
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as e:
            raise FarmError(
                f"reading {p} requires PyYAML. fix: 'pip install PyYAML'"
            ) from e
        loaded = yaml.safe_load(p.read_text())
        return loaded if isinstance(loaded, dict) else None
    return None
