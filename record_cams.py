#!/usr/bin/env python3
"""Record the farm daemon's live camera feeds to H.264 mp4s under recording/.

Polls the running daemon's /v1/cameras/<name>.jpg endpoints (so it never reopens
the RealSense devices the daemon already holds — no libusb contention) and pipes
each JPEG stream into its own ffmpeg encoder. Real-time paced.

  start:  .venv/bin/python record_cams.py [--fps 20] [--cams base,wrist]
  stop:   touch recording/STOP        (clean — finalizes the mp4s)
          or send SIGTERM / Ctrl-C
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime


def fetch(url: str, timeout: float = 2.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                return None
            return r.read()
    except Exception:
        return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    ap.add_argument("--cams", default="base,wrist")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--outdir", default=os.path.join(here, "recording"))
    ap.add_argument("--seconds", type=float, default=0.0, help="max duration; 0 = until stopped")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    stop_file = os.path.join(args.outdir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fps = max(1.0, args.fps)
    period = 1.0 / fps

    procs: dict[str, tuple] = {}   # cam -> (ffmpeg, url, path)
    last: dict[str, bytes] = {}    # cam -> last good jpeg bytes
    for cam in cams:
        url = f"{args.base}/v1/cameras/{cam}.jpg"
        jpg = None
        for _ in range(60):        # wait up to ~6s for the camera to come alive
            jpg = fetch(url)
            if jpg:
                break
            time.sleep(0.1)
        if not jpg:
            print(f"[record] {cam}: no frame from {url} "
                  f"(is the daemon running with --cameras?)", file=sys.stderr)
            continue
        path = os.path.join(args.outdir, f"{cam}_{stamp}.mp4")
        ff = subprocess.Popen(  # noqa: S603,S607
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "image2pipe", "-framerate", f"{fps:g}", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", path],
            stdin=subprocess.PIPE,
        )
        procs[cam] = (ff, url, path)
        last[cam] = jpg
        print(f"[record] {cam} -> {path}")

    if not procs:
        print("[record] no cameras to record; exiting", file=sys.stderr)
        return 2

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))

    print(f"[record] recording {list(procs)} at {fps:g} fps -> {args.outdir}")
    print(f"[record] stop with:  touch {stop_file}   (or kill -TERM {os.getpid()})", flush=True)

    n = 0
    t0 = time.perf_counter()
    next_t = t0
    try:
        while not stop["v"]:
            if os.path.exists(stop_file):
                break
            if args.seconds and (time.perf_counter() - t0) >= args.seconds:
                break
            for cam, (ff, url, _path) in procs.items():
                jpg = fetch(url) or last[cam]   # reuse last frame on a miss → stays real-time
                last[cam] = jpg
                try:
                    ff.stdin.write(jpg)
                except BrokenPipeError:
                    stop["v"] = True
                    break
            n += 1
            if n % int(fps * 5) == 0:
                el = time.perf_counter() - t0
                print(f"[record] {n} frames · {el:.0f}s · {n / el:.1f} fps", flush=True)
            next_t += period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.perf_counter()    # fell behind — resync, don't burst
    finally:
        for _cam, (ff, _url, path) in procs.items():
            try:
                ff.stdin.close()
            except Exception:
                pass
            ff.wait()
            sz = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"[record] saved {path}  ({n} frames, ~{n / fps:.1f}s, {sz / 1e6:.1f} MB)")
        if os.path.exists(stop_file):
            os.remove(stop_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
