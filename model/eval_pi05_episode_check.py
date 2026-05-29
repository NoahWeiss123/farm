"""Sanity-check the live pi 0.5 policy against a recorded LeRobot episode.

Picks frames from one episode's parquet + MP4s, sends each one's
observation to the running serve_policy.py, and compares the predicted
10-step action chunk against the recorded actions ``action[i:i+10]``.

If the model was trained correctly, on a training-set episode the
predicted chunk should be *close* to the recorded chunk — RMSE per
joint << 0.1 rad. Larger errors suggest dataset/transform/checkpoint
mismatch.

Usage::

    python tools/eval_pi05_episode_check.py \\
        --policy-url ws://127.0.0.1:8000 \\
        --episode 0 \\
        --frames 0,30,60,120,200

The eval client (``eval_pi05.py``) is imported for its
``WebSocketPolicy`` so the protocol stays in one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

# Reuse the WebSocketPolicy + ndarray packer from the eval client so the
# wire protocol stays in lock-step.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_pi05 import WebSocketPolicy  # noqa: E402

DATASET_ROOT = HERE.parent / "datasets" / "lerobot" / "farm_uf850_bottle"
JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "grip")


def load_task_prompts() -> dict[int, str]:
    path = DATASET_ROOT / "meta" / "tasks.jsonl"
    out: dict[int, str] = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[int(row["task_index"])] = row["task"]
    return out


def load_episode_parquet(ep_idx: int):
    path = DATASET_ROOT / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    t = pq.read_table(path)
    state = np.array([row.as_py() for row in t["observation.state"]], dtype=np.float32)
    action = np.array([row.as_py() for row in t["action"]], dtype=np.float32)
    task_index = int(t["task_index"][0].as_py())
    return state, action, task_index, t.num_rows


def video_frame(ep_idx: int, cam: str, frame_idx: int) -> np.ndarray:
    """Return frame ``frame_idx`` from ``observation.images.<cam>`` mp4
    as an H×W×3 uint8 RGB array."""
    path = (
        DATASET_ROOT / "videos" / "chunk-000"
        / f"observation.images.{cam}"
        / f"episode_{ep_idx:06d}.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"failed to read {cam} frame {frame_idx} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def resize(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def fmt_row(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in vec) + "]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy-url", default="ws://127.0.0.1:8000")
    ap.add_argument("--episode", type=int, default=0,
                    help="episode_index to load from datasets_lerobot")
    ap.add_argument(
        "--frames", default="0,20,60,120",
        help="comma-separated frame indices to probe",
    )
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument(
        "--override-prompt", default=None,
        help="optional language prompt; default = the episode's recorded task",
    )
    args = ap.parse_args(argv)

    print(f"[check] loading episode {args.episode}")
    state, action, task_index, n_frames = load_episode_parquet(args.episode)
    tasks = load_task_prompts()
    recorded_prompt = tasks.get(task_index, "")
    prompt = args.override_prompt or recorded_prompt
    print(f"[check] frames={n_frames} · state.shape={state.shape} · action.shape={action.shape}")
    print(f"[check] task_index={task_index} · prompt={prompt!r}")

    frame_idxs = [int(x) for x in args.frames.split(",") if x.strip()]
    # Clamp the highest frame so we always have at least 1 ground-truth
    # action to compare against (the parquet has one final padded action).
    frame_idxs = [i for i in frame_idxs if 0 <= i < n_frames]
    if not frame_idxs:
        print("[check] no valid frame indices", file=sys.stderr)
        return 2

    print(f"[check] connecting to {args.policy_url} …")
    policy = WebSocketPolicy(args.policy_url)
    print(
        f"[check] connected · server reports action_dim={policy.action_dim_reported}, "
        f"horizon={policy.action_horizon_reported}"
    )

    # Collected per-frame absolute errors over the model's predicted
    # horizon for the final summary.
    horizon_errs: list[np.ndarray] = []  # each (H, 7)

    for i in frame_idxs:
        base = resize(video_frame(args.episode, "base", i), args.image_size)
        wrist = resize(video_frame(args.episode, "wrist", i), args.image_size)
        obs = {
            "observation/image": base,
            "observation/wrist_image": wrist,
            "observation/state": state[i],
            "prompt": prompt,
        }
        pred = policy.infer(obs)         # (H, 7)
        H = int(pred.shape[0])
        # Ground truth = the recorded actions[i:i+H]; clamp at episode end.
        gt = action[i:i+H]
        compared = min(H, gt.shape[0])
        diff = pred[:compared] - gt[:compared]   # (h, 7) signed errors
        abs_err = np.abs(diff)
        horizon_errs.append(abs_err)

        print()
        print(f"── frame {i} ── compared {compared} of {H} predicted steps ──")
        print(
            "  current state : "
            f"{fmt_row(state[i])}"
        )
        print(
            "  predicted t+0 : "
            f"{fmt_row(pred[0])}"
        )
        print(
            "  recorded  t+0 : "
            f"{fmt_row(action[i])}"
        )
        # Per-dim RMSE across the compared horizon
        rmse = np.sqrt(np.mean(diff ** 2, axis=0))
        per_dim = "  rmse / dim   : " + " ".join(
            f"{n}={v:.4f}" for n, v in zip(JOINT_NAMES, rmse, strict=False)
        )
        print(per_dim)
        mae = abs_err.mean(axis=0)
        per_dim_mae = "  mae  / dim   : " + " ".join(
            f"{n}={v:.4f}" for n, v in zip(JOINT_NAMES, mae, strict=False)
        )
        print(per_dim_mae)

    # Aggregate.
    if horizon_errs:
        # Pad to the shortest horizon so we can stack.
        m = min(e.shape[0] for e in horizon_errs)
        stack = np.stack([e[:m] for e in horizon_errs], axis=0)  # (F, m, 7)
        mae = stack.mean(axis=(0, 1))
        rmse = np.sqrt((stack ** 2).mean(axis=(0, 1)))
        print()
        print(f"=== aggregate over {len(frame_idxs)} frame(s), {m} horizon steps ===")
        print("  MAE  / dim : " + " ".join(f"{n}={v:.4f}" for n, v in zip(JOINT_NAMES, mae, strict=False)))
        print("  RMSE / dim : " + " ".join(f"{n}={v:.4f}" for n, v in zip(JOINT_NAMES, rmse, strict=False)))
        # Crude pass/fail thresholds. Pi 0.5 LoRA on 66 eps shouldn't be
        # perfect, but on training-set frames the joint MAE should be
        # well under 0.1 rad (~6°). Anything > 0.3 rad means something
        # systemic is broken (wrong action mode, wrong key mapping,
        # checkpoint not actually loaded, etc.).
        max_joint_mae = float(mae[:6].max())
        grip_mae = float(mae[6])
        print()
        if max_joint_mae > 0.3:
            print(f"  ⚠  MAX joint MAE {max_joint_mae:.3f} rad — suspicious; "
                  "expect <0.1 on training-set frames")
        elif max_joint_mae > 0.1:
            print(f"  ⚠  MAX joint MAE {max_joint_mae:.3f} rad — higher than ideal "
                  "but possibly fine for diverse training data")
        else:
            print(f"  ✓ joints look sensible (max MAE {max_joint_mae:.3f} rad)")
        if grip_mae > 0.2:
            print(f"  ⚠  gripper MAE {grip_mae:.3f} — high; check binarization")
        else:
            print(f"  ✓ gripper MAE {grip_mae:.3f}")

    policy.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
