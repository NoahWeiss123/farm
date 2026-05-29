"""Two-D435 grabber via per-camera subprocesses.

librealsense + two D435s in one Python process on macOS hard-crashes inside
``wait_for_frames`` with SIGTRAP after a few seconds of normal use. The
previous rig solved this by running a C++ server per camera; we do the same
with a Python module (``farm_edge_agent.cameras.cam_server``) launched as a
subprocess. The main daemon proxies each subprocess's ``/frame.jpg``
endpoint through ``latest(name)``.

This module owns three things:

1. **Device enumeration** — done once, in the parent process, to map
   serials to logical labels (``base`` / ``wrist``).
2. **Subprocess lifecycle** — one ``python -m farm_edge_agent.cameras.cam_server``
   per camera, restarted lazily on failure.
3. **Label swap** — pure relabel (no subprocess restart) so the dashboard
   can flip which feed is which without disturbing USB.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import closing
from typing import Any

import numpy as np

log = logging.getLogger("farm.cameras.realsense")


class RealsenseUnavailable(Exception):
    """Raised when pyrealsense2 isn't importable or no devices are present."""


def _import_rs() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise RealsenseUnavailable(
            "pyrealsense2 not installed — pip install pyrealsense2-macosx (macOS) "
            "or pyrealsense2 (linux/win)"
        ) from e
    return rs


def list_devices() -> list[tuple[str, str]]:
    rs = _import_rs()
    ctx = rs.context()
    return [
        (d.get_info(rs.camera_info.serial_number),
         d.get_info(rs.camera_info.name))
        for d in ctx.query_devices()
    ]


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CamProc:
    """One subprocess running a single D435.

    Restart policy: librealsense on macOS crashes periodically inside
    the USB transfer code — not the user's fault, not avoidable from
    Python. We just respawn forever with a backoff so a permanently
    unplugged camera doesn't peg CPU. The previous "give up after 3
    crashes" cap meant a transient USB hiccup turned the camera feed
    off for the rest of the session, which is worse than the noise.
    """

    # Minimum gap between respawn attempts, scales linearly with the
    # restart counter (clamped at MAX_RESTART_GAP).
    RESTART_BACKOFF_S = 5.0
    MAX_RESTART_GAP_S = 60.0

    # Process-wide guard so two cam subprocesses never enter
    # ``pipeline.start`` concurrently — librealsense on macOS races inside
    # ``claim_interface`` and that race is what the IOKit memory-corruption
    # crash falls out of. Applies to first launch AND every restart.
    _START_LOCK: threading.Lock = threading.Lock()
    _LAST_START_T: float = 0.0
    MIN_INTER_START_GAP_S = 3.0

    # Keep the most-recent good JPEG to serve during transient fetch
    # failures (subprocess HTTP stall, 503 between pipeline restarts,
    # empty-body race at startup). Without this, every blip lets the
    # dashboard fall into its "no signal" path and the tile flickers.
    # Short window: long enough to absorb a sub-frame stall, short
    # enough that a real freeze becomes visible quickly instead of
    # masquerading as a paused live feed.
    STALE_CACHE_S = 2.0

    def __init__(
        self,
        serial: str,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.serial = serial
        self._w = width
        self._h = height
        self._fps = fps
        self._port = _free_port()
        self._proc: subprocess.Popen | None = None
        self._url_frame = f"http://127.0.0.1:{self._port}/frame.jpg"
        self._url_health = f"http://127.0.0.1:{self._port}/healthz"
        self._restarts = 0
        self._last_restart_t = 0.0
        self._last_blob: bytes | None = None
        self._last_blob_t: float = 0.0

    def start(self) -> None:
        argv = [
            sys.executable, "-m", "farm_edge_agent.cameras.cam_server",
            "--serial", self.serial,
            "--port", str(self._port),
            "--width", str(self._w),
            "--height", str(self._h),
            "--fps", str(self._fps),
        ]
        # Pipe stderr/stdout to /dev/null so a chatty subprocess can't fill
        # this process's pipes. The subprocess logs to its own stderr which
        # we discard; if you need to debug, run cam_server by hand.
        with _CamProc._START_LOCK:
            gap = (
                _CamProc._LAST_START_T
                + _CamProc.MIN_INTER_START_GAP_S
                - time.time()
            )
            if gap > 0:
                time.sleep(gap)
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            _CamProc._LAST_START_T = time.time()
        log.info("cam subprocess up: serial=%s port=%d pid=%d",
                 self.serial, self._port, self._proc.pid)

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def latest_jpeg(self) -> bytes | None:
        """Fetch the latest JPEG bytes from the subprocess.

        Short timeout so a wedged subprocess doesn't stall the parent.
        On transient failure (timeout, 503 between pipeline restarts,
        empty-body race during subprocess warmup) returns the last good
        JPEG so the dashboard tile doesn't flicker — but only if the
        cache is recent enough (``STALE_CACHE_S``); past that we report
        the real failure so a dead camera goes visibly black.
        """
        blob: bytes | None
        try:
            with urllib.request.urlopen(self._url_frame, timeout=1.0) as r:
                blob = r.read()
        except Exception:
            blob = None
        # Empty body = subprocess raced; treat the same as a failure.
        if blob:
            self._last_blob = blob
            self._last_blob_t = time.time()
            return blob
        if (
            self._last_blob is not None
            and (time.time() - self._last_blob_t) < self.STALE_CACHE_S
        ):
            return self._last_blob
        return None


class RealsenseGrabber:
    """Two-D435 grabber, one subprocess per camera.

    ``mapping`` is ``{logical_name: serial}``. ``mapping=None`` auto-assigns
    the first enumerated serial to ``base`` and the second to ``wrist``.
    ``swap()`` flips the label → serial assignment in-place without
    restarting either subprocess.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        devs = list_devices()
        if not devs:
            raise RealsenseUnavailable("no RealSense devices connected")
        if mapping is None:
            if len(devs) < 2:
                mapping = {"base": devs[0][0]}
            else:
                mapping = {"base": devs[0][0], "wrist": devs[1][0]}

        # One subprocess per unique serial. Labels point at serials.
        self._procs: dict[str, _CamProc] = {
            serial: _CamProc(serial, width=width, height=height, fps=fps)
            for serial in set(mapping.values())
        }
        self._map_lock = threading.Lock()
        self._label_to_serial: dict[str, str] = dict(mapping)
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    def start(self) -> None:
        # ``_CamProc.start`` itself enforces an inter-start gap process-wide,
        # so we can just fire them off in sequence — first launch and
        # post-crash restart are handled by the same mechanism.
        for p in self._procs.values():
            p.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="cam-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._watchdog_stop.set()
        for p in self._procs.values():
            p.stop()

    def _watchdog_loop(self) -> None:
        # Tight watchdog so a crashed cam comes back fast. The per-proc
        # backoff inside ``reap_dead`` keeps a wedged camera from
        # respawn-flooding USB.
        while not self._watchdog_stop.is_set():
            self.reap_dead()
            self._watchdog_stop.wait(2.0)

    def names(self) -> list[str]:
        with self._map_lock:
            return list(self._label_to_serial)

    def serial(self, name: str) -> str | None:
        with self._map_lock:
            return self._label_to_serial.get(name)

    def latest(self, name: str) -> np.ndarray | None:
        """Return the latest frame as an RGB numpy array, or None.

        Goes via the cam subprocess's HTTP endpoint and JPEG-decodes; this
        is the path consumed by the FARM camera endpoint when it wants to
        re-resize before re-encoding. For pass-through delivery, prefer
        ``latest_jpeg`` (avoids a decode/re-encode round-trip).
        """
        jpeg = self.latest_jpeg(name)
        if jpeg is None:
            return None
        try:
            import cv2
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except ImportError:
            return None

    def latest_jpeg(self, name: str) -> bytes | None:
        """Pass-through fetch of the cam subprocess's latest JPEG bytes."""
        with self._map_lock:
            serial = self._label_to_serial.get(name)
        if serial is None:
            return None
        proc = self._procs.get(serial)
        if proc is None:
            return None
        return proc.latest_jpeg()

    def alive(self) -> dict[str, bool]:
        with self._map_lock:
            mapping = dict(self._label_to_serial)
        out: dict[str, bool] = {}
        for name, serial in mapping.items():
            proc = self._procs.get(serial)
            out[name] = bool(proc and proc.alive())
        return out

    def swap(self) -> dict[str, str]:
        with self._map_lock:
            if set(self._label_to_serial) != {"base", "wrist"}:
                return dict(self._label_to_serial)
            self._label_to_serial = {
                "base": self._label_to_serial["wrist"],
                "wrist": self._label_to_serial["base"],
            }
            return dict(self._label_to_serial)

    # Idle health check the supervisor can call to restart dead subprocesses.
    def reap_dead(self) -> list[str]:
        dead: list[str] = []
        now = time.time()
        for serial, proc in list(self._procs.items()):
            if proc.alive():
                continue
            gap = min(
                proc.RESTART_BACKOFF_S * max(1, proc._restarts),
                proc.MAX_RESTART_GAP_S,
            )
            if (now - proc._last_restart_t) < gap:
                continue
            proc._restarts += 1
            proc._last_restart_t = now
            log.warning(
                "cam subprocess for sn %s died; restart attempt %d",
                serial, proc._restarts,
            )
            proc.start()
            dead.append(serial)
        return dead


_ = time  # imported for future watchdog use; kept to avoid churning imports
