from __future__ import annotations

import json
from pathlib import Path

import pytest
from farm_edge_agent.run_record import RunRecordWriter
from farm_edge_agent.run_record.schema import (
    EVENT_TYPES,
    MAX_EVENT_BYTES,
    ActionChunk,
    ActionChunkData,
    RunStarted,
    RunStartedData,
)


def _run_started(run_id: str = "r_test", ts: float = 1.0) -> RunStarted:
    return RunStarted(
        ts=ts,
        data=RunStartedData(
            run_id=run_id,
            task="pick the red block",
            workspace="my-lab",
            agent_version="0.0.1",
            protocol_version="1.2",
            calibration_hash="sha256:abc",
            config_snapshot={"driver": "lerobot-mock"},
        ),
    )


def _action_chunk(step: int, ts: float) -> ActionChunk:
    return ActionChunk(
        ts=ts,
        data=ActionChunkData(
            node_id="n0",
            chunk_index=step,
            step_index=step,
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            action_space="ee_pose_delta_base_frame",
        ),
    )


def test_writer_appends_valid_jsonl(tmp_path: Path) -> None:
    writer = RunRecordWriter("r_test", root=tmp_path)
    writer.write(_run_started())
    writer.write(_action_chunk(0, ts=1.1))
    writer.write(_action_chunk(1, ts=1.2))
    writer.close()

    lines = (tmp_path / "r_test" / "record.jsonl").read_text().splitlines()
    assert len(lines) == 3
    types_seen = []
    for line in lines:
        payload = json.loads(line)
        kind = payload["type"]
        assert kind in EVENT_TYPES
        EVENT_TYPES[kind].model_validate(payload)
        types_seen.append(kind)
    assert types_seen == ["run_started", "action_chunk", "action_chunk"]


def test_writer_writes_sorted_keys(tmp_path: Path) -> None:
    with RunRecordWriter("r_test", root=tmp_path) as writer:
        writer.write(_run_started())
    line = (tmp_path / "r_test" / "record.jsonl").read_text().splitlines()[0]
    assert line.index('"data"') < line.index('"ts"') < line.index('"type"')


def test_writer_rejects_unknown_event_type(tmp_path: Path) -> None:
    writer = RunRecordWriter("r_test", root=tmp_path)
    with pytest.raises(ValueError, match="unknown event type"):
        writer.write({"ts": 1.0, "type": "nonsense", "data": {}})
    writer.close()


def test_writer_rejects_oversized_event(tmp_path: Path) -> None:
    big = "x" * (MAX_EVENT_BYTES + 100)
    writer = RunRecordWriter("r_test", root=tmp_path)
    payload = _run_started().model_dump(mode="json")
    payload["data"]["task"] = big
    with pytest.raises(ValueError, match=r"> 4096"):
        writer.write(payload)
    writer.close()


def test_writer_appends_across_reopens(tmp_path: Path) -> None:
    with RunRecordWriter("r_test", root=tmp_path) as w:
        w.write(_run_started())
    with RunRecordWriter("r_test", root=tmp_path) as w:
        w.write(_action_chunk(0, ts=1.5))
    lines = (tmp_path / "r_test" / "record.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "run_started"
    assert json.loads(lines[1])["type"] == "action_chunk"
