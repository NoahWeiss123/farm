from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from farm_edge_agent.cli.commands.export import export as export_cmd
from farm_edge_agent.run_record import (
    FarmError,
    RunRecord,
    load_run_record,
    to_lerobot_shards,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_run.jsonl"


def test_exporter_produces_parquet_readable_by_pandas(tmp_path: Path) -> None:
    record = load_run_record(FIXTURE)
    shard = to_lerobot_shards(record, tmp_path)
    assert shard == tmp_path / "data" / "chunk-000" / "episode_000000.parquet"

    df = pd.read_parquet(shard)
    assert list(df.columns) == [
        "episode_index",
        "frame_index",
        "step_index",
        "timestamp",
        "observation.state",
        "observation.ee_pose",
        "action",
        "action_space",
        "task_index",
        "next.done",
    ]
    assert len(df) == 3
    assert df["frame_index"].tolist() == [0, 1, 2]
    assert df["next.done"].tolist() == [False, False, True]
    assert df["timestamp"].iloc[0] == pytest.approx(0.0)
    assert df["timestamp"].iloc[-1] == pytest.approx(0.5)


def test_exporter_writes_meta_files(tmp_path: Path) -> None:
    record = load_run_record(FIXTURE)
    to_lerobot_shards(record, tmp_path)

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["lerobot_version"] == "v0.5"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 3
    assert info["run_id"] == "r_sample_001"
    assert info["calibration_hash"] == "sha256:a8b"

    episodes_line = (tmp_path / "meta" / "episodes.jsonl").read_text().strip()
    episode = json.loads(episodes_line)
    assert episode["episode_index"] == 0
    assert episode["length"] == 3


def test_exporter_byte_stable_on_fixture(tmp_path: Path) -> None:
    record = load_run_record(FIXTURE)
    a = to_lerobot_shards(record, tmp_path / "run_a").read_bytes()
    b = to_lerobot_shards(record, tmp_path / "run_b").read_bytes()
    assert a == b


def test_exporter_raises_without_paired_chunks(tmp_path: Path) -> None:
    record = RunRecord(run_id="r_empty", events=[])
    with pytest.raises(ValueError, match="no paired"):
        to_lerobot_shards(record, tmp_path)


def test_cli_export_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FARM_RUNS_DIR", str(tmp_path))
    run_dir = tmp_path / "r_sample_001"
    run_dir.mkdir()
    (run_dir / "record.jsonl").write_bytes(FIXTURE.read_bytes())

    runner = CliRunner()
    result = runner.invoke(export_cmd, ["r_sample_001"])
    assert result.exit_code == 0, result.output
    export_path = run_dir / "export.jsonl"
    assert export_path.exists()
    assert export_path.read_bytes() == FIXTURE.read_bytes()


def test_cli_export_lerobot_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FARM_RUNS_DIR", str(tmp_path))
    run_dir = tmp_path / "r_sample_001"
    run_dir.mkdir()
    (run_dir / "record.jsonl").write_bytes(FIXTURE.read_bytes())

    runner = CliRunner()
    result = runner.invoke(export_cmd, ["r_sample_001", "--format", "lerobot"])
    assert result.exit_code == 0, result.output
    shard = run_dir / "data" / "chunk-000" / "episode_000000.parquet"
    assert shard.exists()
    table = pq.read_table(shard)
    assert table.num_rows == 3


def test_cli_export_missing_run_raises_farm_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FARM_RUNS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(export_cmd, ["r_missing"], standalone_mode=False)
    assert isinstance(result.exception, FarmError)
    assert "FARM-E4001" in str(result.exception)
