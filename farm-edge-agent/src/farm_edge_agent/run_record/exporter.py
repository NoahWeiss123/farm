"""Run-record → LeRobot v0.5 parquet shard exporter.

The Phase-MVP exporter writes a single LeRobot-shaped episode per run:

    out_dir/
      data/chunk-000/episode_000000.parquet
      meta/info.json
      meta/episodes.jsonl

One row per (obs_chunk, action_chunk) pair sharing the same ``step_index``. Frames
without a matching counterpart are dropped silently — they cannot be exported
because LeRobot frames are obs↔action paired by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from farm_edge_agent.run_record.schema import ActionChunk, ObsChunk, RunRecord, RunStarted

LEROBOT_VERSION = "v0.5"
PARQUET_COMPRESSION = "snappy"


def to_lerobot_shards(record: RunRecord, out_dir: Path) -> Path:
    """Write ``record`` as a LeRobot v0.5 episode shard under ``out_dir``.

    Returns the path of the written parquet file. ``out_dir`` and its
    ``data/chunk-000`` and ``meta`` subdirectories are created if needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data" / "chunk-000"
    meta_dir = out_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    obs_by_step: dict[int, ObsChunk] = {}
    act_by_step: dict[int, ActionChunk] = {}
    started: RunStarted | None = None
    task_label = ""

    for event in record.events:
        if isinstance(event, RunStarted):
            started = event
            task_label = event.data.task
        elif isinstance(event, ObsChunk):
            obs_by_step[event.data.step_index] = event
        elif isinstance(event, ActionChunk):
            act_by_step[event.data.step_index] = event

    paired_steps = sorted(set(obs_by_step) & set(act_by_step))
    if not paired_steps:
        raise ValueError(
            f"run {record.run_id!r} has no paired obs/action chunks to export"
        )

    base_ts = obs_by_step[paired_steps[0]].ts
    rows = _build_rows(paired_steps, obs_by_step, act_by_step, base_ts)
    table = pa.Table.from_pylist(rows, schema=_schema())

    shard_path = data_dir / "episode_000000.parquet"
    pq.write_table(
        table,
        shard_path,
        compression=PARQUET_COMPRESSION,
        use_dictionary=False,
        write_statistics=False,
    )

    _write_meta(meta_dir, record, started, task_label, len(paired_steps))
    return shard_path


def _build_rows(
    paired_steps: list[int],
    obs_by_step: dict[int, ObsChunk],
    act_by_step: dict[int, ActionChunk],
    base_ts: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_idx = len(paired_steps) - 1
    for frame_index, step in enumerate(paired_steps):
        obs = obs_by_step[step]
        act = act_by_step[step]
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame_index,
                "step_index": step,
                "timestamp": float(obs.ts - base_ts),
                "observation.state": list(obs.data.joint_state),
                "observation.ee_pose": list(obs.data.ee_pose),
                "action": list(act.data.action),
                "action_space": act.data.action_space,
                "task_index": 0,
                "next.done": frame_index == last_idx,
            }
        )
    return rows


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("step_index", pa.int64()),
            pa.field("timestamp", pa.float64()),
            pa.field("observation.state", pa.list_(pa.float64())),
            pa.field("observation.ee_pose", pa.list_(pa.float64())),
            pa.field("action", pa.list_(pa.float64())),
            pa.field("action_space", pa.string()),
            pa.field("task_index", pa.int64()),
            pa.field("next.done", pa.bool_()),
        ]
    )


def _write_meta(
    meta_dir: Path,
    record: RunRecord,
    started: RunStarted | None,
    task_label: str,
    frame_count: int,
) -> None:
    info = {
        "lerobot_version": LEROBOT_VERSION,
        "codebase_version": "farm-edge-agent",
        "fps": 0,
        "total_episodes": 1,
        "total_frames": frame_count,
        "chunks_size": 1,
        "run_id": record.run_id,
        "agent_version": started.data.agent_version if started else None,
        "protocol_version": started.data.protocol_version if started else None,
        "calibration_hash": started.data.calibration_hash if started else None,
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    episode = {
        "episode_index": 0,
        "length": frame_count,
        "tasks": [task_label] if task_label else [],
    }
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8"
    )
