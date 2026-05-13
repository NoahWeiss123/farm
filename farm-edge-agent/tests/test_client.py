"""Tests for the public Python API."""

from __future__ import annotations

import asyncio
import builtins
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from farm_edge_agent.client import (
    CapabilityCard,
    Client,
    Event,
    FarmError,
    Run,
    RunSummary,
)


def test_farm_top_level_alias() -> None:
    from farm import Client as ShimClient

    assert ShimClient is Client


class _MockTransport:
    """Records calls and replays a fixed event list."""

    def __init__(self, events: list[Event], run_id: str = "r_test") -> None:
        self._events = events
        self._run_id = run_id
        self.start_calls: list[dict[str, object]] = []

    def start_run(
        self,
        *,
        task: str,
        task_id: str | None,
        backend: str,
        workspace: str | None,
        api_key: str | None,
    ) -> dict[str, object]:
        self.start_calls.append(
            {
                "task": task,
                "task_id": task_id,
                "backend": backend,
                "workspace": workspace,
                "api_key": api_key,
            }
        )
        return {"run_id": self._run_id}

    def stream_events(self, run_id: str) -> Iterator[Event]:
        yield from self._events


def _drain_async(run: Run) -> list[Event]:
    async def go() -> list[Event]:
        return [e async for e in run.events()]

    return asyncio.run(go())


def test_client_run_yields_expected_events() -> None:
    events = [
        Event(type="run_started", ts=1.0, data={"workspace": "test-ws"}),
        Event(type="action_chunk", ts=1.1, data={"action": [0.0, 0.1, 0.0]}),
        Event(type="run_completed", ts=2.0, data={"status": "completed"}),
    ]
    transport = _MockTransport(events)
    client = Client(api_key="test", workspace="test-ws", transport=transport)
    run = client.run("pick the red block", task_id="exp_1", backend="auto")
    assert run.run_id == "r_test"
    assert run.task == "pick the red block"
    assert run.task_id == "exp_1"
    yielded = _drain_async(run)
    assert [e.type for e in yielded] == ["run_started", "action_chunk", "run_completed"]
    assert transport.start_calls[0] == {
        "task": "pick the red block",
        "task_id": "exp_1",
        "backend": "auto",
        "workspace": "test-ws",
        "api_key": "test",
    }


def test_client_run_without_transport_raises() -> None:
    client = Client(api_key="test")
    with pytest.raises(FarmError, match="no transport"):
        client.run("task")


def test_wait_returns_summary_and_caches_events() -> None:
    events = [
        Event(type="run_started", ts=10.0, data={"workspace": "lab"}),
        Event(type="action_chunk", ts=10.5, data={"action": [0.0]}),
        Event(type="run_completed", ts=20.0, data={"status": "completed"}),
    ]
    transport = _MockTransport(events)
    client = Client(api_key="test", transport=transport)
    run = client.run("task")
    summary = run.wait()
    assert isinstance(summary, RunSummary)
    assert summary.status == "completed"
    assert summary.started_at == 10.0
    assert summary.completed_at == 20.0
    assert summary.workspace == "lab"
    # Replay through events() after wait() drains the stream.
    replay = _drain_async(run)
    assert [e.type for e in replay] == ["run_started", "action_chunk", "run_completed"]


def test_to_dataframe_lazy_pandas_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pandas" or name.startswith("pandas."):
            raise ModuleNotFoundError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "pandas", raising=False)

    transport = _MockTransport([Event(type="run_completed", ts=1.0, data={})])
    client = Client(api_key="test", transport=transport)
    run = client.run("task")
    run.wait()
    with pytest.raises(FarmError, match="pandas"):
        run.to_dataframe()


def test_to_dataframe_with_pandas() -> None:
    pd = pytest.importorskip("pandas")
    events = [
        Event(type="run_started", ts=1.0, data={}),
        Event(
            type="action_chunk",
            ts=1.1,
            data={"action": [0.0, 0.1], "obs": {"joint_0": 0.5}},
        ),
        Event(
            type="action_chunk",
            ts=1.2,
            data={"action": [0.1, 0.2], "obs": {"joint_0": 0.6}},
        ),
        Event(type="run_completed", ts=2.0, data={"status": "completed"}),
    ]
    transport = _MockTransport(events)
    client = Client(api_key="test", transport=transport)
    run = client.run("task")
    run.wait()
    df = run.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["ts", "action", "observation.joint_0"]
    assert df["observation.joint_0"].tolist() == [0.5, 0.6]


def _write_record(run_dir: Path, events: list[dict[str, object]]) -> Path:
    run_dir.mkdir(parents=True)
    record = run_dir / "record.jsonl"
    with record.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return record


def test_runs_list_reads_local_fixture(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_record(
        runs_dir / "r_a",
        [
            {
                "type": "run_started",
                "ts": 100.0,
                "data": {"task": "stack", "task_id": "t1", "backend": "auto", "workspace": "lab"},
            },
            {"type": "run_completed", "ts": 200.0, "data": {"status": "completed"}},
        ],
    )
    _write_record(
        runs_dir / "r_b",
        [
            {
                "type": "run_started",
                "ts": 300.0,
                "data": {"task": "wave", "backend": "classical", "workspace": "other"},
            }
        ],
    )
    client = Client(api_key="test", runs_dir=runs_dir)

    all_runs = client.runs.list()
    assert {r.run_id for r in all_runs} == {"r_a", "r_b"}
    assert {r.status for r in all_runs} == {"completed", "running"}

    lab_only = client.runs.list(workspace="lab")
    assert [r.run_id for r in lab_only] == ["r_a"]

    since_run_b = client.runs.list(since="1970-01-01T00:04:00+00:00")
    assert [r.run_id for r in since_run_b] == ["r_b"]


def test_runs_list_missing_dir_returns_empty(tmp_path: Path) -> None:
    client = Client(api_key="test", runs_dir=tmp_path / "does_not_exist")
    assert client.runs.list() == []


def test_runs_export_jsonl_round_trip(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    record = _write_record(
        runs_dir / "r_x",
        [{"type": "run_started", "ts": 1.0, "data": {"task": "t"}}],
    )
    client = Client(api_key="test", runs_dir=runs_dir)
    out = client.runs.export("r_x", format="jsonl")
    assert out == runs_dir / "r_x" / "export.jsonl"
    assert out.read_bytes() == record.read_bytes()


def test_runs_export_missing_raises(tmp_path: Path) -> None:
    client = Client(api_key="test", runs_dir=tmp_path / "runs")
    with pytest.raises(FarmError, match="run not found"):
        client.runs.export("nope")


def test_runs_export_unknown_format_raises(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_record(runs_dir / "r_x", [{"type": "run_started", "ts": 1.0, "data": {}}])
    client = Client(api_key="test", runs_dir=runs_dir)
    with pytest.raises(FarmError, match="unknown export format"):
        client.runs.export("r_x", format="parquet")


def test_cards_list_empty(tmp_path: Path) -> None:
    client = Client(api_key="test", cards_dir=tmp_path / "cards")
    assert client.cards.list() == []


def test_cards_list_json(tmp_path: Path) -> None:
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    (cards_dir / "pi05.json").write_text(
        json.dumps({"id": "pi05-ft", "skills": ["stack", "pick"], "latency_ms": 80})
    )
    client = Client(api_key="test", cards_dir=cards_dir)
    cards = client.cards.list()
    assert len(cards) == 1
    assert isinstance(cards[0], CapabilityCard)
    assert cards[0].id == "pi05-ft"
    assert cards[0].skills == ["stack", "pick"]
    assert cards[0].raw["latency_ms"] == 80


def test_config_path_loads_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("yaml")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("workspace: from-file\napi_key: from-file-key\n")
    monkeypatch.delenv("FARM_API_KEY", raising=False)
    client = Client(config_path=cfg)
    assert client.workspace == "from-file"
    assert client.api_key == "from-file-key"


def test_config_missing_file_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FARM_API_KEY", raising=False)
    monkeypatch.setenv("FARM_HOME", str(tmp_path / "farm_home"))
    client = Client(api_key="explicit")
    assert client.api_key == "explicit"
    assert client.workspace is None
