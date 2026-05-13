from __future__ import annotations

from pathlib import Path

from farm_edge_agent.run_record import (
    ActionChunk,
    RunRecordWriter,
    RunStarted,
    load_run_record,
)
from farm_edge_agent.run_record.schema import (
    ActionChunkData,
    CriticNote,
    CriticNoteData,
    RunStartedData,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_run.jsonl"


def test_reader_round_trips_writer(tmp_path: Path) -> None:
    started = RunStarted(
        ts=1.0,
        data=RunStartedData(
            run_id="r_rt",
            task="pick",
            workspace="my-lab",
            agent_version="0.0.1",
            protocol_version="1.2",
            calibration_hash="sha256:abc",
            config_snapshot={},
        ),
    )
    action = ActionChunk(
        ts=2.0,
        data=ActionChunkData(
            node_id="n0",
            chunk_index=0,
            step_index=0,
            action=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            action_space="ee_pose_delta_base_frame",
        ),
    )
    note = CriticNote(ts=3.0, data=CriticNoteData(node_id="n0", text="ok"))

    with RunRecordWriter("r_rt", root=tmp_path) as w:
        w.write(started)
        w.write(action)
        w.write(note)

    record = load_run_record(tmp_path / "r_rt" / "record.jsonl")
    assert record.run_id == "r_rt"
    assert [e.type for e in record.events] == ["run_started", "action_chunk", "critic_note"]
    assert record.events[0] == started
    assert record.events[1] == action
    assert record.events[2] == note


def test_reader_parses_committed_fixture() -> None:
    record = load_run_record(FIXTURE)
    assert record.run_id == "r_sample_001"
    kinds = [e.type for e in record.events]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_completed"
    assert "safety_event" in kinds
    assert "fallback_invoked" in kinds
    assert "recovery_invoked" in kinds
    assert kinds.count("obs_chunk") == 3
    assert kinds.count("action_chunk") == 3
    started = record.started
    assert started is not None
    assert started.data.calibration_hash == "sha256:a8b"
    assert started.data.protocol_version == "1.2"


def test_reader_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    body = FIXTURE.read_text()
    path.write_text("\n" + body + "\n\n")
    record = load_run_record(path, run_id="r_sample_001")
    assert len(record.events) == len(body.strip().splitlines())
