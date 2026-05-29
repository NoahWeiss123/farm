"""Audit a FARM LeRobot v2.0 dataset for VLA-training quality signals.

Reads the parquet shards under ``<dataset>/data/`` (the output of
``tools/export_lerobot.py``) and reports the things that actually move
π0.5 fine-tune quality:

* action/state alignment — confirms ``action[t] == state[t+1]`` (no off-by-one)
* trajectory smoothness — per-step joint deltas + velocity sign-flip rate
  (a jitter proxy: a smooth human demo flips velocity sign rarely; a jerky
  one flips often). If the *data* is smooth, jitter at inference is a model
  or chunk-stitching artifact, not something the policy learned.
* gripper usage — distribution + grasp band + open/close transitions
* task / episode-length distribution and start/end pose diversity
* (optional) raw-frame timestamp regularity from ``--raw <Dataset dir>``

Nothing here needs a GPU, openpi, or the cluster — it runs on the laptop
against the local export.

    python tools/analyze_dataset.py --dataset datasets_lerobot/farm_uf850_bottle
    python tools/analyze_dataset.py --dataset datasets_lerobot/farm_uf850_bottle --raw Dataset3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
# A joint move below this per-step magnitude counts the frame as "static".
STATIC_EPS_RAD = math.radians(0.05)
# Gripper threshold separating "open" from "grasping" when counting
# transitions. The bottle grasp sits near ~0.3 in this data, so we use a
# low threshold rather than 0.5 (which would miss every grasp).
GRIP_GRASP_THRESHOLD = 0.15


def load_episodes(dataset: Path) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    """Return [(name, state[N,7], action[N,7], task_index), ...]."""
    files = sorted(glob.glob(str(dataset / "data" / "chunk-*" / "episode_*.parquet")))
    if not files:
        sys.exit(f"no parquet shards under {dataset}/data/ — is this a LeRobot export?")
    out = []
    for f in files:
        df = pd.read_parquet(f)
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        ac = np.stack(df["action"].to_numpy()).astype(np.float64)
        ti = int(df["task_index"].iloc[0]) if "task_index" in df else -1
        out.append((os.path.basename(f), st, ac, ti))
    return out


def load_tasks(dataset: Path) -> dict[int, str]:
    path = dataset / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                tasks[int(row["task_index"])] = row["task"]
    return tasks


def report_alignment(eps) -> None:
    max_off = 0.0
    for _, s, a, _ in eps:
        if s.shape[0] > 1:
            max_off = max(max_off, float(np.abs(a[:-1] - s[1:]).max()))
    verdict = "OK" if max_off < 1e-5 else "MISALIGNED — investigate export"
    print(f"\n[action=next-state]  max|action[t]-state[t+1]| = {max_off:.2e}  ({verdict})")


def report_smoothness(eps) -> None:
    d1 = np.concatenate([np.diff(s[:, :6], axis=0) for _, s, _, _ in eps if s.shape[0] > 2])
    print("\n[smoothness]  per-step joint deltas (rad @ dataset fps)")
    for i in range(6):
        col = np.abs(d1[:, i])
        print(f"  {JOINT_NAMES[i]}: |Δ| mean={col.mean():.5f}  p99={np.percentile(col, 99):.5f}  max={col.max():.5f}")
    flips = (np.sign(d1[1:]) != np.sign(d1[:-1])).mean(axis=0)
    static = (np.abs(d1).max(axis=1) < STATIC_EPS_RAD).mean()
    print(f"  velocity sign-flip rate/joint: {np.round(flips, 3)}  (↑ = jittery; <0.1 is smooth)")
    print(f"  static-frame fraction: {static:.3f}")


def report_gripper(eps) -> None:
    g = np.concatenate([s[:, 6] for _, s, _, _ in eps])
    bins = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.01]
    hist = np.histogram(g, bins=bins)[0] / len(g)
    trans = []
    for _, s, _, _ in eps:
        gb = (s[:, 6] > GRIP_GRASP_THRESHOLD).astype(int)
        trans.append(int(np.abs(np.diff(gb)).sum()) if len(gb) > 1 else 0)
    print(f"\n[gripper]  mean={g.mean():.3f}  min={g.min():.3f}  max={g.max():.3f}")
    print(f"  hist [0-.05,.05-.2,.2-.4,.4-.6,.6-.8,.8-1]: {np.round(hist, 3)}")
    print(f"  grasp transitions/ep (thr={GRIP_GRASP_THRESHOLD}): "
          f"mean={np.mean(trans):.2f}  eps_with_0={sum(1 for x in trans if x == 0)}/{len(trans)}")


def report_tasks(eps, tasks: dict[int, str]) -> None:
    lens = [s.shape[0] for _, s, _, _ in eps]
    print(f"\n[episodes]  n={len(eps)}  frames={sum(lens)}  "
          f"len min/mean/max={min(lens)}/{np.mean(lens):.0f}/{max(lens)}")
    by_task: dict[int, int] = {}
    for _, _, _, ti in eps:
        by_task[ti] = by_task.get(ti, 0) + 1
    print(f"[tasks]  {len(by_task)} unique")
    for ti in sorted(by_task):
        print(f"  [{ti}] ×{by_task[ti]:3d}  {tasks.get(ti, '<unknown>')!r}")
    if len(by_task) <= 3:
        print("  ⚠ very few tasks — expect a hard ceiling on out-of-distribution generalization.")
    starts = np.stack([s[0, :6] for _, s, _, _ in eps])
    ends = np.stack([s[-1, :6] for _, s, _, _ in eps])
    print(f"[pose diversity]  start joint std: {np.round(starts.std(0), 3)}")
    print(f"                  end   joint std: {np.round(ends.std(0), 3)}")


def report_raw_timing(raw: Path, limit: int = 80) -> None:
    eps = sorted(glob.glob(str(raw / "episode_*")))[:limit]
    dts = []
    for ep in eps:
        fp = os.path.join(ep, "frames.jsonl")
        if not os.path.isfile(fp):
            continue
        ts = [json.loads(ln)["t"] for ln in open(fp) if ln.strip()]
        if len(ts) > 2:
            dts.append(np.diff(np.array(ts)))
    if not dts:
        print(f"\n[raw timing]  no frames.jsonl under {raw}")
        return
    dt = np.concatenate(dts)
    print(f"\n[raw timing]  {len(eps)} eps · dt mean={dt.mean():.4f}s ({1 / dt.mean():.1f}Hz) "
          f"std={dt.std():.4f} p99={np.percentile(dt, 99):.4f} max={dt.max():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=Path("datasets_lerobot/farm_uf850_bottle"),
                    help="LeRobot v2.0 dataset dir (export_lerobot.py output)")
    ap.add_argument("--raw", type=Path, default=None,
                    help="optional raw episode dir (e.g. Dataset3) for frame-timestamp regularity")
    args = ap.parse_args()

    if not (args.dataset / "data").is_dir():
        sys.exit(f"{args.dataset}/data not found")
    eps = load_episodes(args.dataset)
    tasks = load_tasks(args.dataset)
    print(f"loaded {len(eps)} episodes from {args.dataset}")
    report_alignment(eps)
    report_smoothness(eps)
    report_gripper(eps)
    report_tasks(eps, tasks)
    if args.raw is not None:
        report_raw_timing(args.raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
