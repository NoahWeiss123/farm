"""Offline validation of server-side Real-Time Chunking (RTC).

Replays a recorded episode (raw frames.jsonl + JPEGs) through the served pi0.5
policy and, at several splice points, compares the *next* action chunk generated
WITH RTC guidance against the same chunk generated WITHOUT it. The metric is how
far the new chunk deviates, at the seam, from where the previous chunk said the
trajectory should continue — i.e. the discontinuity the arm would feel when we
swap chunks.

Scenario at each splice frame ``t`` (re-plan stride = ``offset``, inference delay
= ``delay`` steps, horizon ``H``):

    A        = infer(frame t,        reset=True)            # server stores A
    B_rtc    = infer(frame t+offset, offset, delay)         # guided to continue A
    B_plain  = infer(frame t+offset, reset=True)            # independent chunk

The previous chunk A, shifted by ``offset``, predicts the trajectory the arm is
already on. So for k in [0, H-offset):

    dev[k] = || B[k, :6] - A[offset+k, :6] ||     (joint space, degrees)

RTC should drive dev≈0 over the frozen prefix [0, delay) and keep it small over
the soft region, while the plain chunk deviates freely — biggest at the seam
(k=0), which is the jump the arm would jerk through.

Runs on the login pod (reaches the serve node directly), openpi env, PIL for
JPEGs (no ffmpeg needed):

    uv run python offline_check.py /home/nhweiss/farm-train/eval_episode \\
        --host slinky-3 --offset 3 --delay 2
"""
import argparse
import json
import os
import time

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image

JN = ["j1", "j2", "j3", "j4", "j5", "j6"]


def state_of(f):
    return np.array(list(f["joints"]) + [float(f.get("gripper_pos", 0.0))], dtype=np.float32)


def load_img(epdir, cam, i):
    p = os.path.join(epdir, "cameras", cam, f"{i:06d}.jpg")
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(a, 224, 224))


def infer(client, epdir, frames, states, t, task, *, rtc):
    obs = {
        "observation/image": load_img(epdir, "base", t),
        "observation/wrist_image": load_img(epdir, "wrist", t),
        "observation/state": states[t],
        "prompt": task,
    }
    if rtc:
        obs.update(rtc)
    t0 = time.time()
    out = client.infer(obs)
    dt = (time.time() - t0) * 1000
    return np.asarray(out["actions"], dtype=np.float32), dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("epdir", nargs="?", default="/home/nhweiss/farm-train/eval_episode")
    ap.add_argument("--host", default="slinky-3")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--offset", type=int, default=3, help="re-plan stride in steps")
    ap.add_argument("--delay", type=int, default=2, help="inference delay (frozen prefix) in steps")
    ap.add_argument("--n-splices", type=int, default=8)
    args = ap.parse_args()

    H = 10
    overlap = H - args.offset
    meta = json.load(open(os.path.join(args.epdir, "meta.json")))
    task = meta["description"]
    frames = [json.loads(ln) for ln in open(os.path.join(args.epdir, "frames.jsonl")) if ln.strip()]
    states = [state_of(f) for f in frames]
    n = len(states)
    print(f"episode {os.path.basename(args.epdir.rstrip('/'))} | {n} frames | task={task!r}")
    print(f"RTC params: H={H}, offset={args.offset}, delay(frozen)={args.delay}, "
          f"overlap={overlap}  (frozen [0,{args.delay}) · soft [{args.delay},{overlap}))")
    print(f"connecting to {args.host}:{args.port} …")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    splice_ts = np.linspace(5, n - args.offset - H - 1, args.n_splices).astype(int)
    seam_rtc, seam_plain = [], []
    frozen_rtc, frozen_plain = [], []
    soft_rtc, soft_plain = [], []
    lat = []
    first = True
    for t in splice_ts:
        A, dA = infer(client, args.epdir, frames, states, t, task, rtc={"rtc_reset": True})
        B_rtc, dB = infer(client, args.epdir, frames, states, t + args.offset, task,
                          rtc={"rtc_offset": args.offset, "rtc_delay": args.delay})
        B_plain, _ = infer(client, args.epdir, frames, states, t + args.offset, task,
                           rtc={"rtc_reset": True})
        lat += [dA, dB]
        # deviation of each B from A's continuation, in degrees, over the overlap
        dev_rtc = np.degrees(np.linalg.norm(B_rtc[:overlap, :6] - A[args.offset:args.offset + overlap, :6], axis=1))
        dev_plain = np.degrees(np.linalg.norm(B_plain[:overlap, :6] - A[args.offset:args.offset + overlap, :6], axis=1))
        seam_rtc.append(dev_rtc[0])
        seam_plain.append(dev_plain[0])
        frozen_rtc.append(dev_rtc[:args.delay].mean())
        frozen_plain.append(dev_plain[:args.delay].mean())
        if overlap > args.delay:
            soft_rtc.append(dev_rtc[args.delay:].mean())
            soft_plain.append(dev_plain[args.delay:].mean())
        if first:
            first = False
            print(f"  (first infer {dA:.0f} ms incl JIT)")
            print(f"  example @t={t}: per-step deviation from A's continuation (deg):")
            print("     k:       " + " ".join(f"{k:6d}" for k in range(overlap)))
            print("     RTC:     " + " ".join(f"{x:6.2f}" for x in dev_rtc))
            print("     plain:   " + " ".join(f"{x:6.2f}" for x in dev_plain))

    def stat(a):
        a = np.array(a)
        return f"mean {a.mean():6.2f}°  max {a.max():6.2f}°"

    print("\n=== seam discontinuity (k=0: first new action vs where A said to go) ===")
    print(f"  RTC  : {stat(seam_rtc)}")
    print(f"  plain: {stat(seam_plain)}")
    sr, sp = np.array(seam_rtc).mean(), np.array(seam_plain).mean()
    if sp > 1e-6:
        print(f"  → RTC cuts the seam jump by {100*(1-sr/sp):.0f}%  ({sp:.2f}° → {sr:.2f}°)")
    print(f"\n=== frozen-prefix region [0,{args.delay}) (should be ~0 for RTC) ===")
    print(f"  RTC  : {stat(frozen_rtc)}")
    print(f"  plain: {stat(frozen_plain)}")
    if soft_rtc:
        print(f"\n=== soft region [{args.delay},{overlap}) ===")
        print(f"  RTC  : {stat(soft_rtc)}")
        print(f"  plain: {stat(soft_plain)}")
    print(f"\n  median infer latency (intra-cluster): {np.median(lat[2:]):.0f} ms")


if __name__ == "__main__":
    main()
