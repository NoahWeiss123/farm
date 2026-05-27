"""Episode recorder — saves joint trajectories + camera frames to ``datasets/``.

Layout per episode (one directory per take, easy to convert to LeRobot
or HDF5 later for Pi 0.5 / Groot N1 training)::

    datasets/episode_<UTC>_<uuid8>/
        meta.json
        frames.jsonl       # one JSON row per snapshot at fixed FPS
        cameras/<name>/<frame_index:06d>.jpg

Recording is driven by the Quest A / B buttons (start-save / cancel) via
the ROS-TCP bridge. A *single* persistent worker thread services every
episode, because MuJoCo's GL renderers are bound to the thread that
created them — spawning a new thread per ``start()`` left stale
renderer contexts in the sim cache and the second episode's first
render would hang.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("farm.recorder")


class Recorder:
    def __init__(self, supervisor, *, datasets_dir: Path, fps: float = 30.0) -> None:
        self._supervisor = supervisor
        self._datasets_dir = Path(datasets_dir)
        self._datasets_dir.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)

        self._lock = threading.Lock()

        # Episode state — set under ``_lock``, read by the worker
        # before each frame so a cancel can interrupt cleanly.
        self._episode_dir: Path | None = None
        self._episode_id: str = ""
        self._frames_fh = None
        self._frame_count = 0
        self._start_time = 0.0
        self._cameras: list[str] = []
        self._cancelled = False
        # The most-recently-finalized episode's frame count, kept around
        # so the stop_save / cancel HTTP response can report it after
        # the per-episode state has been reset for the next take.
        self._last_finalized_count = 0
        self._last_finalized_path: Path | None = None

        # Worker control signals.
        self._start_event = threading.Event()  # main thread → worker: begin episode
        self._stop_event = threading.Event()   # main thread → worker: end episode
        self._terminate = False
        self._worker = threading.Thread(
            target=self._run_forever, daemon=True, name="farm-recorder"
        )
        self._worker.start()

    # ── public state ──────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._episode_dir is not None

    def is_recording_locked(self) -> bool:
        """Same as is_recording but expects the caller to already hold
        ``self._lock``. Used to avoid double-acquisition inside ``_end``."""
        return self._episode_dir is not None

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_locked()

    # ── start / stop / cancel ─────────────────────────────────────────────

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._episode_dir is not None:
                return {"ok": False, "error": "already recording", **self._state_locked()}
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            short = uuid.uuid4().hex[:8]
            self._episode_id = f"episode_{ts}_{short}"
            self._episode_dir = self._datasets_dir / self._episode_id
            self._episode_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._cameras = list(self._supervisor.cameras())
            except Exception:
                self._cameras = []
            for cam in self._cameras:
                (self._episode_dir / "cameras" / cam).mkdir(parents=True, exist_ok=True)
            self._frames_fh = (self._episode_dir / "frames.jsonl").open("w")
            self._frame_count = 0
            self._start_time = time.time()
            self._cancelled = False
            self._stop_event.clear()
            self._start_event.set()
            log.info("recording started: %s (cameras=%s)", self._episode_id, self._cameras)
            return {"ok": True, **self._state_locked()}

    def stop_save(self) -> dict[str, Any]:
        return self._end(cancel=False)

    def cancel(self) -> dict[str, Any]:
        return self._end(cancel=True)

    def _end(self, *, cancel: bool) -> dict[str, Any]:
        with self._lock:
            if self._episode_dir is None:
                return {"ok": False, "error": "not recording"}
            episode_id = self._episode_id
            self._cancelled = cancel
            self._stop_event.set()

        # Wait (briefly) for the worker to finalize this episode. The
        # worker clears _episode_dir on its way out, so polling that
        # under the lock is the cleanest "are we done?" check.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self._lock:
                if self._episode_dir is None:
                    break
            time.sleep(0.02)

        with self._lock:
            # On the happy path the worker has already produced these
            # diagnostics; otherwise we fall back to the last-known
            # snapshot. ``_last_finalized_*`` survives past the
            # episode-state reset so the response can report meaningful
            # numbers.
            count = (
                self._last_finalized_count if not self.is_recording_locked()
                else self._frame_count
            )
            path = (
                str(self._last_finalized_path)
                if self._last_finalized_path is not None
                else None
            )
            result = {
                "ok": True,
                "episode_id": episode_id,
                "saved": (not cancel),
                "cancelled": cancel,
                "frame_count": count,
                "path": path,
            }
        log.info(
            "recording %s: %s",
            "cancelled" if cancel else "saved",
            episode_id,
        )
        return result

    # ── worker ────────────────────────────────────────────────────────────

    def _run_forever(self) -> None:
        """One worker, many episodes. Single thread keeps the MuJoCo
        renderer cache hot — switching threads between episodes left
        renderers with stale GL contexts that hung on the next render."""
        while not self._terminate:
            # Block until start() arms us.
            if not self._start_event.wait(timeout=1.0):
                continue
            self._start_event.clear()

            # Snapshot the per-episode state up front so the loop body
            # touches local vars rather than thrashing self._lock.
            with self._lock:
                episode_dir = self._episode_dir
                frames_fh = self._frames_fh
                cameras = list(self._cameras)
                start_t = self._start_time
            if episode_dir is None or frames_fh is None:
                continue

            dt = 1.0 / max(1.0, self._fps)
            next_t = time.time()
            frame_idx = 0
            try:
                while not self._stop_event.is_set():
                    now = time.time()
                    sleep_for = next_t - now
                    if sleep_for > 0:
                        if self._stop_event.wait(sleep_for):
                            break
                        continue
                    try:
                        snap = self._supervisor.snapshot()
                    except Exception as exc:
                        log.warning("recorder snapshot failed: %s", exc)
                        next_t += dt
                        continue
                    row = {
                        "frame": frame_idx,
                        "t": round(time.time() - start_t, 4),
                        "joints": snap.get("joints"),
                        "target_joints": snap.get("target_joints"),
                        "gripper_pos": snap.get("gripper_pos"),
                        "gripper": snap.get("gripper"),
                        "tcp_pos_mm": snap.get("tcp_pos_mm"),
                        "tcp_rpy": snap.get("tcp_rpy"),
                        "controller_pose": snap.get("controller_pose"),
                    }
                    try:
                        frames_fh.write(json.dumps(row) + "\n")
                    except Exception as exc:
                        log.warning(
                            "recorder frames write failed at idx %d: %s", frame_idx, exc
                        )
                        break
                    backend = getattr(self._supervisor, "backend", None)
                    cam_jpeg = getattr(backend, "camera_jpeg", None)
                    if callable(cam_jpeg):
                        for cam in cameras:
                            try:
                                blob = cam_jpeg(cam)
                            except Exception as exc:
                                log.warning(
                                    "recorder camera %s frame %d failed: %s",
                                    cam, frame_idx, exc,
                                )
                                continue
                            if blob is None:
                                continue
                            path = episode_dir / "cameras" / cam / f"{frame_idx:06d}.jpg"
                            try:
                                path.write_bytes(blob)
                            except Exception as exc:
                                log.warning(
                                    "recorder camera %s write failed at idx %d: %s",
                                    cam, frame_idx, exc,
                                )
                    frame_idx += 1
                    with self._lock:
                        self._frame_count = frame_idx
                    next_t += dt
                    if next_t < time.time() - dt:
                        next_t = time.time() + dt
            finally:
                # Always finalize, even if we broke out early.
                self._finalize_episode(
                    episode_dir=episode_dir,
                    frames_fh=frames_fh,
                    cameras=cameras,
                    start_t=start_t,
                    frame_count=frame_idx,
                )
                self._stop_event.clear()

    def _finalize_episode(
        self,
        *,
        episode_dir: Path,
        frames_fh,
        cameras: list[str],
        start_t: float,
        frame_count: int,
    ) -> None:
        try:
            frames_fh.flush()
            frames_fh.close()
        except Exception:
            pass
        with self._lock:
            cancelled = self._cancelled
            episode_id = self._episode_id
        duration = time.time() - start_t
        if cancelled:
            try:
                shutil.rmtree(episode_dir)
            except Exception as exc:
                log.warning(
                    "cancelled-episode cleanup failed for %s: %s", episode_dir, exc
                )
        else:
            backend_name = "unknown"
            try:
                backend_name = self._supervisor.backend.backend_name
            except Exception:
                pass
            meta = {
                "episode_id": episode_id,
                "start_iso": datetime.fromtimestamp(
                    start_t, tz=timezone.utc
                ).isoformat(),
                "duration_s": round(duration, 3),
                "fps": self._fps,
                "frame_count": frame_count,
                "cameras": cameras,
                "backend": backend_name,
                "format": "farm-episode-v1",
            }
            try:
                (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            except Exception as exc:
                log.warning("recorder meta write failed: %s", exc)
            log.info(
                "recording saved: %s (%d frames, %.2f s, %s)",
                episode_id,
                frame_count,
                duration,
                str(episode_dir),
            )
        # Reset shared state so a new start() can begin, but stash the
        # final count + path so the (already-blocked) stop_save / cancel
        # caller can include them in the HTTP response.
        with self._lock:
            self._last_finalized_count = frame_count
            self._last_finalized_path = None if cancelled else episode_dir
            self._episode_dir = None
            self._episode_id = ""
            self._frames_fh = None
            self._frame_count = 0
            self._cancelled = False

    # ── helpers ───────────────────────────────────────────────────────────

    def _state_locked(self) -> dict[str, Any]:
        recording = self._episode_dir is not None
        elapsed = (time.time() - self._start_time) if recording else 0.0
        return {
            "recording": recording,
            "episode_id": self._episode_id if recording else None,
            "frame_count": self._frame_count,
            "elapsed_s": round(elapsed, 3),
            "fps": self._fps,
        }

__all__ = ["Recorder"]
