"""`farm export <run-id>` — emit JSONL and (optionally) LeRobot parquet shards."""

from __future__ import annotations

import shutil
from pathlib import Path

import click

from farm_edge_agent.run_record import FarmError, load_run_record, to_lerobot_shards
from farm_edge_agent.run_record.writer import _runs_root


@click.command("export", help="Download run record as JSONL + LeRobot shards.")
@click.argument("run_id")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "lerobot"]),
    default="jsonl",
    help="Export format. jsonl always written; 'lerobot' adds parquet shards.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (default: the run dir itself).",
)
def export(run_id: str, fmt: str, out_dir: Path | None) -> None:
    run_dir = _runs_root() / run_id
    record_path = run_dir / "record.jsonl"
    if not record_path.exists():
        raise FarmError("FARM-E4001", f"run {run_id!r} not found at {record_path}")

    target = out_dir if out_dir is not None else run_dir
    target.mkdir(parents=True, exist_ok=True)

    export_jsonl = target / "export.jsonl"
    shutil.copyfile(record_path, export_jsonl)
    click.echo(str(export_jsonl))

    if fmt == "lerobot":
        record = load_run_record(record_path, run_id=run_id)
        shard = to_lerobot_shards(record, target)
        click.echo(str(shard))
