"""Offline validation: replay a recorded episode (raw frames.jsonl + JPEGs)
through the served π0.5 policy and compare predicted actions to the
ground-truth recorded actions. Open-loop, no robot.

Runs on the login pod (reaches the serve node directly) in the openpi env;
uses PIL for JPEGs so no ffmpeg/torchcodec is needed.

  uv run python /home/nhweiss/farm-train/offline_eval.py /home/nhweiss/farm-train/eval_episode
"""
import json
import os
import sys
import time

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image

EPDIR = sys.argv[1] if len(sys.argv) > 1 else "/home/nhweiss/farm-train/eval_episode"
HOST, PORT = "slinky-1", 8000
HORIZON = 10
N_SAMPLES = 16
JN = ["j1", "j2", "j3", "j4", "j5", "j6"]


def state_of(f):
    return np.array(list(f["joints"]) + [float(f.get("gripper_pos", 0.0))], dtype=np.float32)


def load_img(cam, i):
    p = os.path.join(EPDIR, "cameras", cam, f"{i:06d}.jpg")
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(a, 224, 224))


def main():
    meta = json.load(open(os.path.join(EPDIR, "meta.json")))
    task = meta["description"]
    frames = [json.loads(ln) for ln in open(os.path.join(EPDIR, "frames.jsonl")) if ln.strip()]
    states = [state_of(f) for f in frames]
    n = len(states)
    print(f"episode: {os.path.basename(EPDIR.rstrip('/'))} | {n} frames | task={task!r}")
    print(f"connecting to policy server {HOST}:{PORT} …")
    client = websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)

    idxs = np.linspace(0, n - HORIZON - 1, N_SAMPLES).astype(int)
    err0, errH, grip0, tinf = [], [], [], []
    for ni, t in enumerate(idxs):
        obs = {
            "observation/image": load_img("base", t),
            "observation/wrist_image": load_img("wrist", t),
            "observation/state": states[t],
            "prompt": task,
        }
        t0 = time.time()
        pred = np.asarray(client.infer(obs)["actions"], dtype=np.float32)   # (H, 7)
        tinf.append(time.time() - t0)
        gt = np.stack([states[t + 1 + k] for k in range(HORIZON)])          # recorded next states
        err0.append(np.abs(pred[0, :6] - gt[0, :6]))
        errH.append(np.abs(pred[:, :6] - gt[:, :6]).mean(axis=0))
        grip0.append(abs(float(pred[0, 6]) - float(gt[0, 6])))
        if ni == 0:
            print(f"  (first infer {tinf[0]:.1f}s incl. one-time JIT compile)")
            print(f"  @frame {t}: pred[0] joints = {np.round(pred[0, :6], 3)}")
            print(f"             gt   joints = {np.round(gt[0, :6], 3)}  gripper pred={pred[0,6]:.3f} gt={gt[0,6]:.3f}")

    err0 = np.array(err0)
    errH = np.array(errH)
    grip0 = np.array(grip0)
    print(f"\n=== immediate next-step accuracy (pred action[0] vs recorded), {N_SAMPLES} frames ===")
    for j in range(6):
        m = err0[:, j].mean()
        print(f"  {JN[j]}: MAE {m:.4f} rad ({np.degrees(m):5.2f}°)")
    om = err0.mean()
    print(f"  -> overall joint MAE: {om:.4f} rad ({np.degrees(om):.2f}°)")
    print(f"  -> gripper MAE: {grip0.mean():.4f}  (0=open .. ~0.3=closed in this data)")
    print(f"\n=== full {HORIZON}-step chunk vs recorded chunk ===")
    print(f"  joint MAE over chunk: {errH.mean():.4f} rad ({np.degrees(errH.mean()):.2f}°)")
    print(f"\n  median inference latency: {np.median(tinf[1:]) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
