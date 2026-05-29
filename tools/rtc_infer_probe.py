"""Probe the SERVED Policy.infer RTC plumbing (no new GPU job — hits the running
serve). Mirrors the unit test but through the websocket/infer path:

  A       = infer(obs_t, rtc_reset=True)              # server stores A
  B_rtc   = infer(obs_t, rtc_offset=0, rtc_delay=H)   # full-freeze guide -> A
  B_plain = infer(obs_t, rtc_reset=True)              # plain, different noise

Same obs for all three (only the server's RNG differs). With offset=0 + full
freeze, working guidance pins B_rtc onto A regardless of noise. We compare in
OUTPUT (joint, deg) space. If B_rtc ≪ B_plain in deviation from A → infer()
plumbing works; if B_rtc ≈ B_plain → the guided branch isn't firing.
"""
import argparse
import json
import os

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image


def state_of(f):
    return np.array(list(f["joints"]) + [float(f.get("gripper_pos", 0.0))], dtype=np.float32)


def load_img(epdir, cam, i):
    p = os.path.join(epdir, "cameras", cam, f"{i:06d}.jpg")
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(a, 224, 224))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epdir")
    ap.add_argument("--host", default="slinky-3")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    H = 10
    meta = json.load(open(os.path.join(args.epdir, "meta.json")))
    task = meta["description"]
    frames = [json.loads(ln) for ln in open(os.path.join(args.epdir, "frames.jsonl")) if ln.strip()]
    states = [state_of(f) for f in frames]
    c = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    def obs_at(t, rtc):
        o = {
            "observation/image": load_img(args.epdir, "base", t),
            "observation/wrist_image": load_img(args.epdir, "wrist", t),
            "observation/state": states[t],
            "prompt": task,
        }
        o.update(rtc)
        return np.asarray(c.infer(o)["actions"], np.float32)

    rtc_dev, plain_dev = [], []
    ts = np.linspace(5, len(states) - H - 2, args.n).astype(int)
    for t in ts:
        A = obs_at(t, {"rtc_reset": True})
        Brtc = obs_at(t, {"rtc_offset": 0, "rtc_delay": H})   # full freeze -> A
        Bpln = obs_at(t, {"rtc_reset": True})
        rtc_dev.append(np.degrees(np.linalg.norm(Brtc[:, :6] - A[:, :6], axis=1)).mean())
        plain_dev.append(np.degrees(np.linalg.norm(Bpln[:, :6] - A[:, :6], axis=1)).mean())
    rtc_dev, plain_dev = np.array(rtc_dev), np.array(plain_dev)
    print(f"served infer() RTC probe (offset=0, full freeze), {args.n} frames:")
    print(f"  B_plain deviation from A : mean {plain_dev.mean():.3f}°  (sampling-noise floor)")
    print(f"  B_rtc   deviation from A : mean {rtc_dev.mean():.3f}°")
    if plain_dev.mean() > 1e-6:
        print(f"  → guidance reduces deviation by {100*(1-rtc_dev.mean()/plain_dev.mean()):.0f}%")
    print("VERDICT:", "infer() RTC WORKS" if rtc_dev.mean() < 0.6 * plain_dev.mean()
          else "infer() RTC NOT FIRING (plumbing bug)")


if __name__ == "__main__":
    main()
