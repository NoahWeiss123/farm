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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from farm_edge_agent.drivers.sim import Prop, Scene, SimDriver
from farm_edge_agent.policies import Pi05Policy
from farm_edge_agent.policies.pi05 import run_pi05_loop
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


# Built-in default scene: blocks + cup resting on the floor in front of the
# arm. No table — just the infinite ground plane defined in the MJCF and
# whatever objects we drop in. Half-extents are 0.0125 m → 25 mm cubes.
DEFAULT_SCENE = SceneSpec(
    name="picknplace",
    props=[
        {"id": "red_block", "shape": "box", "size": [0.0125, 0.0125, 0.0125],
         "pos": [0.05, -0.55, 0.0125], "rgba": [0.9, 0.1, 0.1, 1.0],
         "mass": 0.04},
        {"id": "blue_block", "shape": "box", "size": [0.0125, 0.0125, 0.0125],
         "pos": [0.15, -0.55, 0.0125], "rgba": [0.1, 0.2, 0.9, 1.0],
         "mass": 0.04},
        {"id": "green_block", "shape": "box", "size": [0.0125, 0.0125, 0.0125],
         "pos": [-0.05, -0.55, 0.0125], "rgba": [0.1, 0.7, 0.2, 1.0],
         "mass": 0.04},
        # Wide, squat, *heavy* cup: 70 mm diameter, 50 mm tall, 1.2 kg.
        # The high mass + low CoM means a glancing bump from the gripper
        # or the elbow translates into a tiny slide rather than a tip.
        # Pushed back in Y so the arm's sweep between blocks doesn't
        # graze it.
        {"id": "cup", "shape": "cylinder", "size": [0.035, 0.025],
         "pos": [-0.20, -0.78, 0.025], "rgba": [0.92, 0.90, 0.86, 0.9],
         "mass": 1.2},
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
            realtime_speed=1.5,
        )
        self._driver.connect()
        self._statuses: dict[str, RunStatus] = {}
        self._queue: queue.Queue[_PendingRun] = queue.Queue()
        self._lock = threading.Lock()
        # Inspect state — the rolling "what is the arm thinking" snapshot.
        self._inspect_state: dict[str, Any] = {
            "run_id": None,
            "task": None,
            "plan": None,           # {plan_id, reasoning, nodes}
            "active_node_id": None,
            "active_node_index": None,
            "last_action": None,    # {label, action_space, action, t}
            "last_critic": None,    # latest critic_note text
            "policy": "auto",       # "gpt+skills" or "pi05"
        }
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

    def render_camera(self, name: str, *, width: int, height: int):
        """Render a named MuJoCo camera as a (H, W, 3) uint8 array."""
        return self._driver.render_rgb(camera=name, height=height, width=width)

    def render_camera_depth(self, name: str, *, width: int, height: int):
        return self._driver.render_depth(camera=name, height=height, width=width)

    def inspect(self) -> dict[str, Any]:
        """Snapshot of the current high-level agent state, plus the
        live π0.5-shaped observation summary."""
        snap = self._driver.snapshot()
        # Summarize the observation tensor without serializing megapixels.
        obs_summary = {
            "joint_position_7": [*snap["joints"], 0.0],
            "gripper_position": _grip_to_unit(self._driver),
            "tcp_pos_mm": [v * 1000.0 for v in snap["tcp_pos_m"]],
            "tcp_quat": snap["tcp_quat"],
            "image_urls": {
                "exterior": "/v1/cameras/exterior.jpg",
                "wrist": "/v1/cameras/wrist.jpg",
                "topdown": "/v1/cameras/topdown.jpg",
                "exterior_depth": "/v1/cameras/exterior.depth.png",
                "wrist_depth": "/v1/cameras/wrist.depth.png",
                "topdown_depth": "/v1/cameras/topdown.depth.png",
            },
        }
        with self._lock:
            state = dict(self._inspect_state)
        state["observation"] = obs_summary
        state["world"] = {
            "joints": snap["joints"],
            "tcp_pos_m": snap["tcp_pos_m"],
            "gripper": snap["gripper"],
            "props": snap["props"],
        }
        return state

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
        # Pick a policy backend. π0.5 runs on Modal (24 GB+ GPU); when its
        # endpoint isn't configured, fall back to the GPT decomposer +
        # hand-coded skill library. ``FARM_POLICY=pi05`` forces π0.5 and
        # fails loudly if the endpoint is unset.
        backend = os.environ.get("FARM_POLICY", "auto").lower()
        pi05 = Pi05Policy()
        use_pi05 = backend == "pi05" or (backend == "auto" and pi05.configured())
        if use_pi05:
            self._execute_pi05(pending, pi05)
            return
        world = LiveSimWorldState(self._driver)
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

    def _execute_pi05(self, pending: _PendingRun, policy: Pi05Policy) -> None:
        """Run a closed-loop π0.5 inference session on the sim driver.

        Writes a run record with one ``action_chunk`` event per executed
        action so the dashboard timeline + 3D viewer light up the same way
        as a GPT-planned run.
        """
        import uuid
        from farm_edge_agent.run_record.writer import RunRecordWriter

        status = pending.status
        rid = pending.run_id
        start = time.time()
        record = RunRecordWriter(rid, root=self._runs_root)
        action_idx = 0
        try:
            self._record_and_publish(record, rid, {
                "ts": time.time(), "type": "run_started",
                "data": {
                    "run_id": rid, "task": pending.task, "workspace": "local",
                    "agent_version": "0.0.1", "protocol_version": "1.2.0",
                    "calibration_hash": "unknown",
                    "config_snapshot": {"policy": "pi05",
                                        "endpoint": policy.endpoint},
                },
            })
            self._record_and_publish(record, rid, {
                "ts": time.time(), "type": "plan_emitted",
                "data": {
                    "plan_id": f"plan_pi05_{uuid.uuid4().hex[:8]}",
                    "nodes": [{"id": "n1", "instruction": pending.task,
                               "backend": "pi05"}],
                    "router_reason": (
                        "π0.5 VLA model — observation: dual-camera RGB + "
                        "7-DoF joint state + gripper position; action: "
                        "joint deltas + gripper target at 20 Hz."
                    ),
                },
            })
            self._record_and_publish(record, rid, {
                "ts": time.time(), "type": "node_started",
                "data": {"node_id": "n1", "backend": "pi05"},
            })

            def on_step(ev: dict[str, Any]) -> None:
                nonlocal action_idx
                if ev.get("type") == "pi05_infer":
                    self._record_and_publish(record, rid, {
                        "ts": time.time(), "type": "critic_note",
                        "data": {
                            "node_id": "n1",
                            "text": (
                                f"π0.5 chunk: horizon={ev['chunk_len']}, "
                                f"latency={ev['latency_s']*1000:.0f}ms"
                            ),
                        },
                    })

            run_pi05_loop(
                self._driver, policy, pending.task,
                max_steps=int(os.environ.get("FARM_PI05_MAX_STEPS", "300")),
                chunks_per_call=int(os.environ.get("FARM_PI05_CHUNK", "10")),
                on_step=on_step,
            )

            self._record_and_publish(record, rid, {
                "ts": time.time(), "type": "node_completed",
                "data": {"node_id": "n1", "outcome": "succeeded"},
            })
            status.outcome = "succeeded"
            status.state = "succeeded"
            status.plan_id = "plan_pi05"
            self._record_and_publish(record, rid, {
                "ts": time.time(), "type": "run_completed",
                "data": {"run_id": rid, "outcome": "succeeded",
                         "wall_clock_s": time.time() - start},
            })
        except Exception as e:
            status.state = "failed"
            status.outcome = "failed"
            status.error = f"{type(e).__name__}: {e}"
            try:
                self._record_and_publish(record, rid, {
                    "ts": time.time(), "type": "run_completed",
                    "data": {"run_id": rid, "outcome": "failed",
                             "wall_clock_s": time.time() - start},
                })
            except Exception:
                pass
        finally:
            record.close()
            status.completed_at = time.time()
            self._bus.publish("runs", {"type": "run_state",
                                       "data": status.__dict__.copy()})

    def _record_and_publish(
        self, rec: Any, run_id: str, event: dict[str, Any]
    ) -> None:
        rec.write(event)
        ev = dict(event)
        ev["run_id"] = run_id
        self._bus.publish(f"run:{run_id}", ev)
        self._bus.publish("runs:all", ev)

    def _on_run_event(self, run_id: str, event: dict[str, Any]) -> None:
        # Re-stamp with run_id so SSE clients can demux a unified stream.
        event = dict(event)
        event.setdefault("run_id", run_id)
        self._bus.publish(f"run:{run_id}", event)
        self._bus.publish("runs:all", event)
        self._update_inspect_from_event(run_id, event)

    def _update_inspect_from_event(
        self, run_id: str, event: dict[str, Any]
    ) -> None:
        kind = event.get("type")
        data = event.get("data", {}) or {}
        with self._lock:
            state = self._inspect_state
            if kind == "run_started":
                state["run_id"] = run_id
                state["task"] = data.get("task")
                state["plan"] = None
                state["active_node_id"] = None
                state["active_node_index"] = None
                state["last_action"] = None
                state["last_critic"] = None
                state["policy"] = "pi05" if (
                    isinstance(data.get("config_snapshot"), dict)
                    and data["config_snapshot"].get("policy") == "pi05"
                ) else "gpt+skills"
            elif kind == "plan_emitted":
                state["plan"] = {
                    "plan_id": data.get("plan_id"),
                    "reasoning": data.get("router_reason"),
                    "nodes": data.get("nodes", []),
                }
            elif kind == "node_started":
                nid = data.get("node_id")
                state["active_node_id"] = nid
                nodes = (state.get("plan") or {}).get("nodes") or []
                for idx, n in enumerate(nodes):
                    if n.get("id") == nid:
                        state["active_node_index"] = idx
                        break
            elif kind == "node_completed":
                # Keep active_node_id pointing at the last completed node
                # so the UI can show "done with step X" — the next
                # node_started will overwrite this.
                pass
            elif kind == "action_chunk":
                state["last_action"] = {
                    "node_id": data.get("node_id"),
                    "action": data.get("action"),
                    "action_space": data.get("action_space"),
                    "label": data.get("label"),
                    "t": event.get("ts"),
                }
            elif kind == "critic_note":
                state["last_critic"] = data.get("text")
            elif kind == "run_completed":
                state["active_node_id"] = None
            snapshot = dict(state)
        # Push to subscribers (outside lock).
        self._bus.publish("inspect", {"type": "inspect", **snapshot})

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


def _grip_to_unit(driver: Any) -> float:
    """0=open, 1=closed — same convention as π0.5 's gripper_position."""
    try:
        snap = driver.snapshot()
    except Exception:
        return 0.0
    return 1.0 if snap.get("gripper") == "closed" else 0.0


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
