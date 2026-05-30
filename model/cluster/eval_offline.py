#!/usr/bin/env python3
"""Offline action-accuracy eval for a trained π0.5 policy — in-process, no robot.

Replays recorder-format episodes (``meta.json`` + ``frames.jsonl`` +
``cameras/{base,wrist}/*.jpg``) through the policy and compares its predicted
action chunk to the recorded ground truth. The recorder stores no explicit
action column — actions are absolute joint positions, so ``action[t]`` is just
``state[t+1]``; the policy un-deltas its joint output back to absolute, so we
compare ``policy.infer(obs)["actions"]`` against the recorded next states.

Loads the checkpoint directly (``create_trained_policy``) — no serve_policy /
websocket. Run it in the openpi container (it needs JAX + a GPU):

  uv run python eval_offline.py \\
      --config pi05_farm_uf850_gse \\
      --checkpoint-dir ~/farm-train/openpi/checkpoints/pi05_farm_uf850_gse/<exp>/2999 \\
      --episodes-dir ~/farm-train/eval_episodes \\
      --n-episodes 6 --samples-per-episode 16 --horizon 10

Reports per-joint MAE (rad + deg), gripper MAE, and full-chunk MAE — per
episode and aggregated. On TRAINING episodes this measures fit/memorization
(can the policy reproduce demos it saw); large error here means training itself
underfit, before generalization is even in question.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from openpi_client import image_tools
from PIL import Image

from openpi.policies import policy_config
from openpi.training import config as _config

JN = ["j1", "j2", "j3", "j4", "j5", "j6"]


def state_of(frame: dict) -> np.ndarray:
    """[6 joints, gripper] → (7,) float32, matching the trained action layout."""
    return np.array(list(frame["joints"]) + [float(frame.get("gripper_pos", 0.0))], dtype=np.float32)


def load_img(epdir: str, cam: str, i: int) -> np.ndarray:
    """Read frame i of a camera, resized+padded to 224×224 uint8 (as in training)."""
    p = os.path.join(epdir, "cameras", cam, f"{i:06d}.jpg")
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(a, 224, 224))


def eval_episode(policy, epdir: str, n_samples: int, horizon: int) -> dict | None:
    meta = json.load(open(os.path.join(epdir, "meta.json")))
    task = meta["description"]
    frames = [json.loads(ln) for ln in open(os.path.join(epdir, "frames.jsonl")) if ln.strip()]
    states = [state_of(f) for f in frames]
    n = len(states)
    if n <= horizon + 1:
        return None
    idxs = np.linspace(0, n - horizon - 1, min(n_samples, n - horizon - 1)).astype(int)
    err0, errH, grip0 = [], [], []
    for t in idxs:
        obs = {
            "observation/image": load_img(epdir, "base", t),
            "observation/wrist_image": load_img(epdir, "wrist", t),
            "observation/state": states[t],
            "prompt": task,
        }
        pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)  # (H, 7) absolute
        gt = np.stack([states[t + 1 + k] for k in range(horizon)])         # recorded next states
        err0.append(np.abs(pred[0, :6] - gt[0, :6]))
        errH.append(np.abs(pred[:, :6] - gt[:, :6]).mean(axis=0))
        grip0.append(abs(float(pred[0, 6]) - float(gt[0, 6])))
    return {"task": task, "n": n, "err0": np.array(err0),
            "errH": np.array(errH), "grip0": np.array(grip0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="registered openpi TrainConfig name")
    ap.add_argument("--checkpoint-dir", required=True, help="dir containing params/ + assets/")
    ap.add_argument("--episodes-dir", required=True, help="parent dir of episode_* recordings")
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--samples-per-episode", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    cfg = _config.get_config(args.config)
    print(f">>> loading policy: config={args.config}\n    ckpt={args.checkpoint_dir}", flush=True)
    t0 = time.time()
    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    print(f"    policy ready in {time.time() - t0:.0f}s", flush=True)

    epdirs = sorted(glob.glob(os.path.join(args.episodes_dir, "episode_*")))[: args.n_episodes]
    if not epdirs:
        print(f"!!! no episode_* dirs under {args.episodes_dir}")
        return
    print(f">>> evaluating {len(epdirs)} episode(s), {args.samples_per_episode} frames each\n", flush=True)

    all0, allH, allg = [], [], []
    for ep in epdirs:
        r = eval_episode(policy, ep, args.samples_per_episode, args.horizon)
        if r is None:
            print(f"  {os.path.basename(ep)}: too short, skipped")
            continue
        m0 = r["err0"].mean()
        print(f"  {os.path.basename(ep)} ({r['n']} fr · {r['task'][:34]!r}): "
              f"joint MAE {m0:.4f} rad ({np.degrees(m0):.2f}°), grip {r['grip0'].mean():.3f}", flush=True)
        all0.append(r["err0"])
        allH.append(r["errH"])
        allg.append(r["grip0"])

    if not all0:
        print("no episodes evaluated")
        return
    e0 = np.concatenate(all0)
    eH = np.concatenate(allH)
    g = np.concatenate(allg)
    print(f"\n=== AGGREGATE · {len(all0)} episodes · {len(e0)} sampled frames ===")
    print("immediate next-step action (pred[0] vs recorded):")
    for j in range(6):
        m = e0[:, j].mean()
        print(f"  {JN[j]}: MAE {m:.4f} rad ({np.degrees(m):5.2f}°)")
    print(f"  -> overall joint MAE: {e0.mean():.4f} rad ({np.degrees(e0.mean()):.2f}°)")
    print(f"  -> gripper MAE: {g.mean():.4f}  (0=open .. ~0.3=closed)")
    print(f"  -> full {args.horizon}-step chunk joint MAE: {eH.mean():.4f} rad ({np.degrees(eH.mean()):.2f}°)")


if __name__ == "__main__":
    main()
