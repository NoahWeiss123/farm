"""RunSupervisor — manages a singleton sim driver + spawns RunLoops on demand.

The HTTP layer turns an incoming POST /v1/runs into a `start_run(task)`
call here. We run the RunLoop in a daemon thread (the loop is sync-CPU
bound on the sim, no async benefit) and publish every event through the
EventBus so SSE subscribers see it live.

Only one run executes at a time on the sim. Concurrent submissions are
queued (FIFO) so the dashboard can fire-and-forget.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from farm_edge_agent.drivers.sim import Prop, Scene, SimDriver
from farm_edge_agent.run_loop import RunLoop, RunSummary
from farm_edge_agent.run_record.writer import DEFAULT_RUNS_ROOT
from farm_edge_agent.safety.factory import make_sim_enforcer
from farm_edge_agent.server.bus import EventBus
from farm_edge_agent.skills import StubPlanner
from farm_edge_agent.skills.gpt_planner import GptPlanner
from farm_edge_agent.skills.library import LiveSimWorldState, SkillExecutor


def _runs_root() -> Path:
    override = os.environ.get("FARM_RUNS_DIR")
    return Path(override) if override else DEFAULT_RUNS_ROOT


@dataclass
class RunStatus:
    run_id: str
    task: str
    state: str  # "queued" | "running" | "succeeded" | "failed" | "aborted_safety"
    submitted_at: float
    started_at: float | None = None
    completed_at: float | None = None
    outcome: str | None = None
    error: str | None = None
    plan_id: str | None = None
    safety_events: int = 0


@dataclass
class _PendingRun:
    task: str
    run_id: str
    status: RunStatus


@dataclass
class SceneSpec:
    name: str
    props: list[dict[str, Any]] = field(default_factory=list)

    def to_scene(self) -> Scene:
        return Scene(
            name=self.name,
            props=[
                Prop(
                    id=p["id"],
                    shape=p["shape"],
                    size=tuple(p["size"]),
                    pos=tuple(p["pos"]),
                    rgba=tuple(p.get("rgba", [0.8, 0.8, 0.8, 1.0])),
                    mass=float(p.get("mass", 0.05)),
                )
                for p in self.props
            ],
        )


# Built-in default scene used for the demo when no /v1/scenes call has
# been made. Mirrors the table+blocks+cup that the pick-and-place
# integration tests use.
DEFAULT_SCENE = SceneSpec(
    name="picknplace",
    props=[
        {"id": "red_block", "shape": "box", "size": [0.0125, 0.0125, 0.0125],
         "pos": [0.00, -0.70, 0.2775], "rgba": [0.9, 0.1, 0.1, 1.0]},
        {"id": "blue_block", "shape": "box", "size": [0.0125, 0.0125, 0.0125],
         "pos": [0.10, -0.70, 0.2775], "rgba": [0.1, 0.2, 0.9, 1.0]},
        {"id": "green_block", "shape": "box", "size": [0.0125, 0.0125, 0.0125],
         "pos": [-0.10, -0.70, 0.2775], "rgba": [0.1, 0.7, 0.2, 1.0]},
        {"id": "cup", "shape": "cylinder", "size": [0.04, 0.04],
         "pos": [-0.12, -0.78, 0.305], "rgba": [0.85, 0.85, 0.85, 0.9]},
    ],
)


class RunSupervisor:
    def __init__(
        self,
        bus: EventBus,
        *,
        scene: SceneSpec | None = None,
        runs_root: Path | None = None,
    ) -> None:
        self._bus = bus
        self._scene_spec = scene or DEFAULT_SCENE
        self._runs_root = runs_root or _runs_root()
        self._driver = SimDriver(
            scene=self._scene_spec.to_scene(),
            event_sink=self._on_driver_event,
            realtime=True,
            # Wall-clock pacing so the dashboard's 3D viewer can keep up.
            # Bump to 2.0 if demos feel too slow.
            realtime_speed=1.0,
        )
        self._driver.connect()
        self._statuses: dict[str, RunStatus] = {}
        self._queue: queue.Queue[_PendingRun] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self._world_thread = threading.Thread(target=self._world_pump, daemon=True)
        self._world_thread.start()

    # ── public API ───────────────────────────────────────────────────────────

    def submit_run(self, task: str, run_id: str | None = None) -> RunStatus:
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        status = RunStatus(
            run_id=rid,
            task=task,
            state="queued",
            submitted_at=time.time(),
        )
        with self._lock:
            self._statuses[rid] = status
        self._queue.put(_PendingRun(task=task, run_id=rid, status=status))
        return status

    def get_run(self, run_id: str) -> RunStatus | None:
        with self._lock:
            return self._statuses.get(run_id)

    def list_runs(self) -> list[RunStatus]:
        with self._lock:
            statuses = list(self._statuses.values())
        # Augment with on-disk runs we never spawned this process
        seen = {s.run_id for s in statuses}
        if self._runs_root.exists():
            for d in sorted(self._runs_root.iterdir(), reverse=True):
                if not d.is_dir() or d.name in seen:
                    continue
                rec = d / "record.jsonl"
                if not rec.exists():
                    continue
                task, outcome = _peek_record(rec)
                statuses.append(
                    RunStatus(
                        run_id=d.name,
                        task=task,
                        state=outcome or "unknown",
                        submitted_at=d.stat().st_mtime,
                        started_at=d.stat().st_mtime,
                        completed_at=d.stat().st_mtime,
                        outcome=outcome,
                    )
                )
        return statuses

    def replay_run(self, run_id: str) -> list[dict[str, Any]]:
        path = self._runs_root / run_id / "record.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events

    def snapshot_world(self) -> dict[str, Any]:
        snap = self._driver.snapshot()
        return {
            "joints": snap["joints"],
            "tcp_pos_m": snap["tcp_pos_m"],
            "tcp_quat": snap["tcp_quat"],
            "gripper": snap["gripper"],
            "props": snap["props"],
            "scene": self._scene_spec.name,
        }

    def scene_spec(self) -> dict[str, Any]:
        return {
            "name": self._scene_spec.name,
            "props": [dict(p) for p in self._scene_spec.props],
        }

    def _planner_context(self) -> dict[str, Any]:
        """Snapshot of scene + live prop positions for the GPT planner."""
        snap = self._driver.snapshot()
        live_props = []
        for p in self._scene_spec.props:
            pid = p["id"]
            entry = snap.get("props", {}).get(pid)
            live_props.append(
                {
                    "id": pid,
                    "shape": p["shape"],
                    "size_m": list(p["size"]),
                    "pos_m": (
                        list(entry["pos"]) if entry else list(p["pos"])
                    ),
                    "rgba": list(p.get("rgba", [0.8, 0.8, 0.8, 1.0])),
                }
            )
        return {
            "scene_name": self._scene_spec.name,
            "workspace_envelope_m": {
                "x": [-0.40, 0.40],
                "y": [-1.05, -0.10],
                "z": [0.265, 0.80],
            },
            "props": live_props,
            "arm": {
                "model": "ufactory_850",
                "gripper": "parallel_jaw",
                "tcp_pose_m": list(snap["tcp_pos_m"]),
                "gripper_state": snap["gripper"],
            },
        }

    # ── worker thread ────────────────────────────────────────────────────────

    def _run_worker(self) -> None:
        while True:
            pending = self._queue.get()
            self._execute(pending)

    def _execute(self, pending: _PendingRun) -> None:
        status = pending.status
        status.state = "running"
        status.started_at = time.time()
        self._bus.publish("runs", {"type": "run_state", "data": status.__dict__.copy()})
        world = LiveSimWorldState(self._driver)
        # GPT decomposer with on-disk plan cache (Layer-1 skill compiler).
        # Falls back to the english parser on OPENAI_API_KEY-less runs.
        planner: Any
        if os.environ.get("OPENAI_API_KEY"):
            planner = GptPlanner(scene_provider=self._planner_context)
        else:
            planner = StubPlanner()
        executor = SkillExecutor(world)
        safety = make_sim_enforcer(self._driver)
        loop = RunLoop(
            driver=self._driver,
            planner=planner,
            executor=executor,
            safety=safety,
            event_sink=lambda e, rid=pending.run_id: self._on_run_event(rid, e),
            runs_root=self._runs_root,
        )
        try:
            summary: RunSummary = loop.run(pending.task, run_id=pending.run_id)
            status.outcome = summary.outcome
            status.state = summary.outcome
            status.error = summary.error
            status.plan_id = summary.plan_id
            status.safety_events = summary.safety_events
        except Exception as e:
            status.state = "failed"
            status.outcome = "failed"
            status.error = f"{type(e).__name__}: {e}"
        status.completed_at = time.time()
        self._bus.publish("runs", {"type": "run_state", "data": status.__dict__.copy()})

    def _on_run_event(self, run_id: str, event: dict[str, Any]) -> None:
        # Re-stamp with run_id so SSE clients can demux a unified stream.
        event = dict(event)
        event.setdefault("run_id", run_id)
        self._bus.publish(f"run:{run_id}", event)
        self._bus.publish("runs:all", event)

    def _on_driver_event(self, event: dict[str, Any]) -> None:
        # Broadcast joint states + grip events to world subscribers.
        if event.get("type") == "joint_state":
            self._bus.publish("world", event)

    def _world_pump(self) -> None:
        """Heartbeat publisher — broadcasts the current world snapshot every
        200 ms so a fresh UI client sees the arm even before a run starts."""
        while True:
            try:
                snap = self.snapshot_world()
                self._bus.publish(
                    "world",
                    {"type": "world_snapshot", **snap, "t": time.time()},
                )
            except Exception:
                pass
            time.sleep(0.2)


def _peek_record(path: Path) -> tuple[str, str | None]:
    """Return (task, outcome) extracted from an on-disk JSONL run record."""
    task = ""
    outcome = None
    try:
        with path.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") == "run_started":
                    task = e.get("data", {}).get("task", "")
                elif e.get("type") == "run_completed":
                    outcome = e.get("data", {}).get("outcome")
    except OSError:
        pass
    return task, outcome


__all__ = ["DEFAULT_SCENE", "RunStatus", "RunSupervisor", "SceneSpec"]
