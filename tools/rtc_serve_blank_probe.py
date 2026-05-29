"""Confirm the DEPLOYED serve fires RTC guidance, using blank (high-uncertainty)
images so consecutive chunks genuinely disagree and the guidance effect is
visible. Hits the running serve over websocket — no GPU job.

  A       = infer(blank, rtc_reset=True)
  B_rtc   = infer(blank, rtc_offset=0, rtc_delay=10)   # full freeze -> A
  B_plain = infer(blank, rtc_reset=True)
"""
import argparse

import numpy as np
from openpi_client import websocket_client_policy

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="slinky-3")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
c = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)


def call(rtc):
    o = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros((7,), np.float32),
        "prompt": "Picking up the bottle and placing it on the box",
    }
    o.update(rtc)
    return np.asarray(c.infer(o)["actions"], np.float32)


rtc_dev, plain_dev = [], []
for _ in range(a.n):
    A = call({"rtc_reset": True})
    Br = call({"rtc_offset": 0, "rtc_delay": 10})
    Bp = call({"rtc_reset": True})
    rtc_dev.append(np.degrees(np.linalg.norm((Br - A)[:, :6], axis=1)).mean())
    plain_dev.append(np.degrees(np.linalg.norm((Bp - A)[:, :6], axis=1)).mean())
rtc_dev, plain_dev = np.array(rtc_dev), np.array(plain_dev)
print(f"DEPLOYED serve RTC probe (blank imgs, offset=0 full freeze), {a.n} frames:")
print(f"  B_plain dev from A: {plain_dev.mean():.2f}°   B_rtc dev from A: {rtc_dev.mean():.2f}°")
print(f"  → guidance reduces deviation {100*(1-rtc_dev.mean()/max(1e-9,plain_dev.mean())):.0f}%")
print("VERDICT:", "serve fires RTC ✓" if rtc_dev.mean() < 0.7 * plain_dev.mean() else "serve NOT firing RTC")
