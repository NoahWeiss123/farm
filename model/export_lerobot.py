"""Convert FARM ``datasets/`` to a LeRobot v2.0 dataset.

Input layout (per episode dir matching ``episode_*``)::

    datasets/episode_<UTC>_<id>/
        meta.json
        frames.jsonl
        cameras/<name>/<frame_idx:06d>.jpg

Output layout (LeRobot v2.0)::

    <out>/
        meta/{info.json, episodes.jsonl, tasks.jsonl, stats.json}
        data/chunk-000/episode_NNNNNN.parquet
        videos/chunk-000/observation.images.<cam>/episode_NNNNNN.mp4

Features:
    observation.state  — float32[7]  (6 joints rad + gripper_pos 0..1)
    action             — float32[7]  (next observation.state; last frame repeats)
    observation.images.base   — video (h264 mp4 @ fps)
    observation.images.wrist  — video (h264 mp4 @ fps)
    timestamp, frame_index, episode_index, task_index, index — scalars

Run::

    python model/export_lerobot.py \\
        --src datasets/dataset3 \\
        --out datasets/lerobot/farm_uf850_bottle \\
        --fps 30
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 1000      # episodes per data chunk dir
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


def discover_episodes(src: Path) -> list[Path]:
    eps = []
    for p in sorted(src.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("episode_"):
            continue
        if not (p / "meta.json").is_file() or not (p / "frames.jsonl").is_file():
            print(f"  skip {p.name}: missing meta or frames", file=sys.stderr)
            continue
        eps.append(p)
    return eps


def load_frames(ep: Path) -> list[dict[str, Any]]:
    with (ep / "frames.jsonl").open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_state(row: dict[str, Any]) -> list[float]:
    joints = row.get("joints") or [0.0] * 6
    grip = row.get("gripper_pos")
    if grip is None or not isinstance(grip, (int, float)):
        grip = 0.0
    out = [float(joints[i]) if i < len(joints) else 0.0 for i in range(6)]
    out.append(float(grip))
    return out


def encode_video(jpegs_dir: Path, out_mp4: Path, fps: int) -> int:
    """ffmpeg image-sequence → h264 mp4. Returns the number of frames encoded."""
    jpegs = sorted(jpegs_dir.glob("*.jpg"))
    if not jpegs:
        raise RuntimeError(f"no jpegs in {jpegs_dir}")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(jpegs_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "medium",
        # LeRobot's video reader expects all frames present; force CFR.
        "-vsync", "cfr", "-r", str(fps),
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return len(jpegs)


def compute_stats(arr: np.ndarray) -> dict[str, Any]:
    """Per-channel stats for a (N, D) float array — matches LeRobot's shape."""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "min": mn.astype(np.float32).tolist(),
        "max": mx.astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
    }


def scalar_stats(arr: np.ndarray) -> dict[str, Any]:
    return {
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "count": [int(arr.shape[0])],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("datasets/dataset3"))
    ap.add_argument("--out", type=Path, default=Path("datasets/lerobot/farm_uf850_bottle"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--robot-type", default="uf850")
    ap.add_argument("--force", action="store_true",
                    help="wipe the output dir first")
    args = ap.parse_args()

    src: Path = args.src
    out: Path = args.out
    if out.exists() and args.force:
        shutil.rmtree(out)
    if out.exists():
        print(f"output dir {out} already exists; pass --force to wipe", file=sys.stderr)
        return 1
    out.mkdir(parents=True)
    (out / "meta").mkdir()
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "videos" / "chunk-000").mkdir(parents=True)

    episodes = discover_episodes(src)
    if not episodes:
        print(f"no episodes under {src}", file=sys.stderr)
        return 1
    print(f"converting {len(episodes)} episodes → {out}")

    # First pass: assign task indices from unique descriptions.
    task_index: dict[str, int] = OrderedDict()
    for ep in episodes:
        meta = json.loads((ep / "meta.json").read_text())
        desc = (meta.get("description") or "").strip()
        if not desc:
            print(f"  WARN {ep.name}: no description, will skip", file=sys.stderr)
            continue
        if desc not in task_index:
            task_index[desc] = len(task_index)

    # Filter out episodes without descriptions.
    episodes = [ep for ep in episodes if
                (json.loads((ep / "meta.json").read_text()).get("description") or "").strip()]

    tasks_path = out / "meta" / "tasks.jsonl"
    with tasks_path.open("w") as fh:
        for desc, idx in task_index.items():
            fh.write(json.dumps({"task_index": idx, "task": desc}) + "\n")

    # Second pass: per-episode parquet + videos.
    episodes_meta: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_timestamps: list[float] = []
    global_index = 0
    cam_keys = ("base", "wrist")
    total_frames = 0

    for ep_idx, ep in enumerate(episodes):
        t0 = time.time()
        meta = json.loads((ep / "meta.json").read_text())
        desc = (meta.get("description") or "").strip()
        task_idx = task_index[desc]
        frames = load_frames(ep)
        # Build state matrix.
        states = np.array([build_state(r) for r in frames], dtype=np.float32)
        # Action = next-state. For the last frame, repeat the final state so
        # the chunk has the same length as the observation chunk.
        actions = np.empty_like(states)
        actions[:-1] = states[1:]
        actions[-1] = states[-1]
        # LeRobot validates that consecutive timestamps are spaced exactly
        # 1/fps within a tight tolerance, so use the frame-index grid (i/fps)
        # rather than the jittery wall-clock capture times in `t` (teleop
        # recording isn't perfectly periodic). Matches LeRobot's fixed-fps
        # convention; the recorded `t` jitter is absorbed.
        timestamps = np.array([i / args.fps for i in range(len(frames))],
                              dtype=np.float32)
        frame_idx_col = np.arange(len(frames), dtype=np.int64)
        ep_idx_col = np.full(len(frames), ep_idx, dtype=np.int64)
        task_idx_col = np.full(len(frames), task_idx, dtype=np.int64)
        global_idx_col = np.arange(global_index, global_index + len(frames), dtype=np.int64)
        global_index += len(frames)

        df = pd.DataFrame({
            "observation.state": [s.tolist() for s in states],
            "action": [a.tolist() for a in actions],
            "timestamp": timestamps,
            "frame_index": frame_idx_col,
            "episode_index": ep_idx_col,
            "task_index": task_idx_col,
            "index": global_idx_col,
        })
        ep_parquet = out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(df), ep_parquet)

        # Encode videos.
        for cam in cam_keys:
            cdir = ep / "cameras" / cam
            if not cdir.is_dir():
                continue
            video_path = (
                out / "videos" / "chunk-000"
                / f"observation.images.{cam}"
                / f"episode_{ep_idx:06d}.mp4"
            )
            encode_video(cdir, video_path, args.fps)

        all_states.append(states)
        all_actions.append(actions)
        all_timestamps.extend(timestamps.tolist())
        total_frames += len(frames)
        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [desc],
            "length": len(frames),
        })
        dt = time.time() - t0
        print(f"  [{ep_idx + 1:3d}/{len(episodes)}] {ep.name}  ·  {len(frames)} fr  ·  {dt:.1f}s")

    # episodes.jsonl
    eps_path = out / "meta" / "episodes.jsonl"
    with eps_path.open("w") as fh:
        for row in episodes_meta:
            fh.write(json.dumps(row) + "\n")

    # stats.json
    state_arr = np.concatenate(all_states, axis=0)
    action_arr = np.concatenate(all_actions, axis=0)
    ts_arr = np.array(all_timestamps, dtype=np.float32)
    stats = {
        "observation.state": compute_stats(state_arr),
        "action": compute_stats(action_arr),
        "timestamp": scalar_stats(ts_arr),
        "frame_index": scalar_stats(np.arange(total_frames, dtype=np.int64)),
        "episode_index": scalar_stats(
            np.concatenate([np.full(e["length"], e["episode_index"]) for e in episodes_meta])
        ),
        "task_index": scalar_stats(
            np.concatenate([
                np.full(e["length"], task_index[e["tasks"][0]]) for e in episodes_meta
            ])
        ),
        "index": scalar_stats(np.arange(total_frames, dtype=np.int64)),
    }
    # Video stats: dummy per-channel (LeRobot expects shape (3,1,1) for HWC images).
    # We compute by sampling the first frame of every episode rather than every
    # pixel — cheap and good enough for the policy normalizer.
    for cam in cam_keys:
        samples = []
        for ep in episodes:
            f0 = ep / "cameras" / cam / "000000.jpg"
            if not f0.is_file():
                continue
            import cv2
            img = cv2.imread(str(f0), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            samples.append(img)
        if not samples:
            continue
        stack = np.stack(samples, axis=0)        # (N, H, W, 3)
        mean = stack.mean(axis=(0, 1, 2))        # (3,)
        std = stack.std(axis=(0, 1, 2))          # (3,)
        mn = stack.min(axis=(0, 1, 2))
        mx = stack.max(axis=(0, 1, 2))
        # LeRobot wants shape (3, 1, 1) for CHW images. Reshape each scalar
        # array to (3,1,1).
        def _r(a: np.ndarray) -> list[list[list[float]]]:
            return a.reshape(3, 1, 1).astype(np.float32).tolist()
        stats[f"observation.images.{cam}"] = {
            "mean": _r(mean),
            "std": _r(std),
            "min": _r(mn),
            "max": _r(mx),
            "count": [int(total_frames)],
        }

    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    # info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": args.robot_type,
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(task_index),
        "total_videos": len(episodes_meta) * len(cam_keys),
        "total_chunks": 1 + (len(episodes_meta) - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(STATE_NAMES)],
                "names": list(STATE_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": [len(STATE_NAMES)],
                "names": list(STATE_NAMES),
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    for cam in cam_keys:
        info["features"][f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": args.fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nwrote {len(episodes_meta)} episodes · {total_frames} frames")
    print(f"  tasks    : {len(task_index)}")
    print(f"  videos   : {len(episodes_meta) * len(cam_keys)} files in {out / 'videos' / 'chunk-000'}")
    print(f"  parquet  : {len(episodes_meta)} files in {out / 'data' / 'chunk-000'}")
    print(f"  meta     : {out / 'meta'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
