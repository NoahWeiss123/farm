"""Single-D435 subprocess server.

Runs as ``python -m farm_edge_agent.cameras.cam_server --serial XYZ --port N``.
Opens *one* RealSense D435 by serial, captures color frames in a daemon
thread, and serves the latest as JPEG over loopback HTTP at ``/frame.jpg``.

This exists because pyrealsense2 + two D435s in one Python process on macOS
hard-crashes inside ``librealsense::pipeline::wait_for_frames`` with SIGTRAP
after a few seconds of joint use. The previous rig used a C++ server per
camera for the same reason ("constrealsense on macOS is unreliable driving
two D435s from a single process, so they're kept in separate processes
entirely"). This is the Python equivalent — the FARM daemon spawns one
subprocess per camera and proxies JPEGs from each.

Endpoints
---------
GET /frame.jpg   latest JPEG
GET /healthz     {"ok": True}
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger("farm.cam_server")


class _Grabber:
    """Single-pipeline producer. Mirrors the threaded shape of the in-process
    grabber but here it owns the whole process."""

    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        self.serial = serial
        self._w = width
        self._h = height
        self._fps = fps
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"d435-{serial}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[bytes | None, float]:
        with self._lock:
            return self._jpeg, self._stamp

    def _run(self) -> None:
        import numpy as np
        import pyrealsense2 as rs
        from PIL import Image

        backoff = 1.0
        while not self._stop.is_set():
            pipeline: Any = None
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(self.serial)
                cfg.enable_stream(
                    rs.stream.color, self._w, self._h, rs.format.rgb8, self._fps
                )
                profile = pipeline.start(cfg)
                # Kill the time_diff_keeper polling thread inside librealsense:
                # on macOS it issues USB control transfers concurrently with
                # the streaming pipeline and races claim_interface, which
                # surfaces as "memory corruption of free block" in libmalloc.
                # We don't need host-side clock-sync for dashboard tiles.
                try:
                    for sensor in profile.get_device().sensors:
                        if sensor.supports(rs.option.global_time_enabled):
                            sensor.set_option(rs.option.global_time_enabled, 0.0)
                except Exception as exc:  # noqa: BLE001
                    log.debug("global_time disable skipped: %s", exc)
                log.info(
                    "streaming %dx%d@%d (sn %s)",
                    self._w, self._h, self._fps, self.serial,
                )
                backoff = 1.0
                while not self._stop.is_set():
                    frames = pipeline.wait_for_frames(2000)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    # IMPORTANT: copy out of the librealsense buffer before
                    # the next iteration recycles it. JPEG-encode here so
                    # the HTTP handler only ever touches plain bytes.
                    arr = np.array(color.get_data(), copy=True)
                    buf = io.BytesIO()
                    Image.fromarray(arr).save(buf, format="JPEG", quality=82)
                    blob = buf.getvalue()
                    with self._lock:
                        self._jpeg = blob
                        self._stamp = time.time()
            except Exception as exc:  # noqa: BLE001 — never let the loop die
                log.warning("rs loop error: %s; reconnecting in %.1fs", exc, backoff)
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            self._stop.wait(backoff)
            backoff = min(8.0, backoff * 1.5)


def _make_handler(grabber: _Grabber):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # silence per-request noise
            pass

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/frame.jpg":
                jpeg, _stamp = grabber.latest()
                if jpeg is None:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            self.send_error(404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Single-D435 subprocess server.")
    ap.add_argument("--serial", required=True, help="D435 serial number")
    ap.add_argument("--port", type=int, required=True, help="loopback HTTP port")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[cam {args.serial[-4:]}] %(message)s",
    )
    grabber = _Grabber(args.serial, args.width, args.height, args.fps)
    grabber.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(grabber))
    log.info("cam server up on 127.0.0.1:%d", args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        grabber.stop()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
