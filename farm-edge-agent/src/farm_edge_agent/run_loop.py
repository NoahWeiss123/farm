"""RunLoop — orchestrates a single run from prompt to completion.

Glues the safety enforcer and recovery primitives onto a Driver and a
pluggable PlanExecutor. The supervisor's `farm start` command and the
`farm run` CLI both go through this loop.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from farm_edge_agent.drivers.base import Driver, Pose
from farm_edge_agent.recovery.primitives import Registry as RecoveryRegistry
from farm_edge_agent.safety import Pose as SafetyPose
from farm_edge_agent.safety import SafetyEvent
from farm_edge_agent.safety.enforcer import SafetyEnforcer

EventSink = Callable[[dict[str, Any]], None]


@dataclass
class PlanNode:
    id: str
    instruction: str
    backend: str = "sim"


@dataclass
class Plan:
    plan_id: str
    nodes: list[PlanNode]
    reasoning: str = ""


class Planner(Protocol):
    def plan(self, task: str, *, run_id: str) -> Plan: ...


class PlanExecutor(Protocol):
    def execute(
        self,
        node: PlanNode,
        driver: Driver,
        run_id: str,
    ) -> ExecResult: ...


@dataclass
class ExecResult:
    ok: bool
    chunks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunSummary:
    run_id: str
    outcome: str
    wall_clock_s: float
    plan_id: str | None
    safety_events: int
    recovery_events: int
    error: str | None = None


class _ChunkCounter:
    def __init__(self) -> None:
        self.action = 0
        self.obs = 0


class RunLoop:
    def __init__(
        self,
        *,
        driver: Driver,
        planner: Planner,
        executor: PlanExecutor,
        safety: SafetyEnforcer,
        recovery: RecoveryRegistry | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self._driver = driver
        self._planner = planner
        self._executor = executor
        self._safety = safety
        self._recovery = recovery or RecoveryRegistry()
        self._event_sink = event_sink

    def run(self, task: str, run_id: str | None = None) -> RunSummary:
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        start = time.time()
        safety_events = 0
        recovery_events = 0
        plan_id: str | None = None
        error: str | None = None
        outcome = "succeeded"

        self._emit({
            "ts": time.time(), "type": "run_started",
            "data": {"run_id": rid, "task": task},
        })

        pre = self._safety.pre_run()
        for ev in pre.events:
            self._emit_safety(ev, node_id=None)
            safety_events += 1
        if not pre.ok:
            outcome = "aborted_safety"
            error = "pre-run safety gate failed"
        else:
            try:
                plan = self._planner.plan(task, run_id=rid)
                plan_id = plan.plan_id
                self._emit({
                    "ts": time.time(), "type": "plan_emitted",
                    "data": {
                        "plan_id": plan.plan_id,
                        "nodes": [
                            {"id": n.id, "instruction": n.instruction, "backend": n.backend}
                            for n in plan.nodes
                        ],
                        "router_reason": plan.reasoning or None,
                    },
                })
                for node in plan.nodes:
                    nstate = self._execute_node(node, rid)
                    safety_events += nstate["safety"]
                    recovery_events += nstate["recovery"]
                    if nstate["outcome"] != "succeeded":
                        outcome = nstate["outcome"]
                        error = nstate.get("error")
                        break
            except Exception as e:
                outcome = "failed"
                error = f"{type(e).__name__}: {e}"

        self._emit({
            "ts": time.time(), "type": "run_completed",
            "data": {"run_id": rid, "outcome": outcome, "wall_clock_s": time.time() - start},
        })

        return RunSummary(
            run_id=rid, outcome=outcome, wall_clock_s=time.time() - start,
            plan_id=plan_id, safety_events=safety_events,
            recovery_events=recovery_events, error=error,
        )

    def _execute_node(self, node: PlanNode, run_id: str) -> dict[str, Any]:
        counter = _ChunkCounter()
        safety_count = 0
        recovery_count = 0
        outcome = "succeeded"
        error: str | None = None

        self._emit({
            "ts": time.time(), "type": "node_started",
            "data": {"node_id": node.id, "backend": node.backend},
        })

        try:
            result = self._executor.execute(node, self._driver, run_id)
        except Exception as e:
            outcome = "failed"
            error = f"executor raised {type(e).__name__}: {e}"
            self._emit({
                "ts": time.time(), "type": "node_completed",
                "data": {"node_id": node.id, "outcome": outcome},
            })
            return {"outcome": outcome, "error": error, "safety": 0, "recovery": 0}

        for chunk in result.chunks:
            kind = chunk.get("type", "action_chunk")
            if kind == "action_chunk":
                safety_check = self._gate_chunk(chunk)
                for ev in safety_check["events"]:
                    self._emit_safety(ev, node_id=node.id)
                    safety_count += 1
                if not safety_check["ok"]:
                    chain = chunk.get("recovery_chain", ["abort_safely"])
                    for primitive in chain:
                        try:
                            self._invoke_recovery(primitive, node.id)
                            recovery_count += 1
                        except Exception as e:
                            outcome = "failed"
                            error = f"recovery {primitive} raised {type(e).__name__}: {e}"
                            break
                    if outcome != "failed":
                        outcome = "aborted_safety"
                        error = "safety violation; recovery completed"
                    break
                self._dispatch_action(chunk)
                action = chunk.get("action", [])
                if chunk.get("action_space") == "gripper":
                    encoded: list[float] = [_encode_gripper(action[0])] if action else []
                else:
                    encoded = [float(a) for a in action]
                self._emit({
                    "ts": time.time(), "type": "action_chunk",
                    "data": {
                        "node_id": node.id, "chunk_index": counter.action,
                        "step_index": counter.action, "action": encoded,
                        "action_space": chunk.get("action_space", "tcp_xyzrpy_mm"),
                        "label": chunk.get("label"),
                    },
                })
                counter.action += 1
                obs = self._snapshot_obs()
                self._emit({
                    "ts": time.time(), "type": "obs_chunk",
                    "data": {
                        "node_id": node.id, "chunk_index": counter.obs,
                        "step_index": counter.obs, "joint_state": obs["joint_state"],
                        "ee_pose": obs["ee_pose"], "image_paths": {},
                    },
                })
                counter.obs += 1
            elif kind == "critic_note":
                self._emit({
                    "ts": time.time(), "type": "critic_note",
                    "data": {"node_id": node.id, "text": str(chunk.get("text", ""))},
                })

        if not result.ok and outcome == "succeeded":
            outcome = "failed"
            error = result.error or "executor returned ok=False"

        self._emit({
            "ts": time.time(), "type": "node_completed",
            "data": {"node_id": node.id, "outcome": outcome},
        })
        return {"outcome": outcome, "error": error, "safety": safety_count, "recovery": recovery_count}

    def _gate_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        from farm_edge_agent.safety import ActionChunk as SafetyActionChunk

        action = chunk.get("action", [])
        waypoints: list[SafetyPose] = []
        if chunk.get("action_space") == "tcp_xyzrpy_mm" and len(action) == 6:
            waypoints.append(SafetyPose(
                x=action[0] / 1000.0, y=action[1] / 1000.0, z=action[2] / 1000.0,
                rx=action[3], ry=action[4], rz=action[5],
            ))
        sac = SafetyActionChunk(
            joint_positions=[], joint_velocities=[],
            tcp_waypoints=waypoints, duration_s=float(chunk.get("duration_s", 0.0)),
        )
        result = self._safety.check_chunk(sac)
        return {"ok": result.ok, "events": result.events}

    def _dispatch_action(self, chunk: dict[str, Any]) -> None:
        space = chunk.get("action_space", "tcp_xyzrpy_mm")
        action = chunk.get("action", [])
        if space == "tcp_xyzrpy_mm" and len(action) == 6:
            pose: Pose = (
                float(action[0]), float(action[1]), float(action[2]),
                float(action[3]), float(action[4]), float(action[5]),
            )
            cap = float(chunk.get("velocity_cap", 100.0))
            self._driver.move_to(pose, cap)
        elif space == "gripper":
            self._driver.set_gripper(action[0] if action else "open")
        else:
            raise ValueError(f"unknown action_space: {space!r}")

    def _snapshot_obs(self) -> dict[str, Any]:
        return {
            "joint_state": list(self._driver.read_joint_state()),
            "ee_pose": list(self._driver.read_tcp_pose()),
        }

    def _invoke_recovery(self, primitive: str, node_id: str) -> None:
        from farm_edge_agent.recovery.primitives import (
            abort_safely,
            home,
            open_gripper,
            retry_grasp,
        )
        fn = self._recovery.get(primitive)
        shim = _SafetyShim()
        if fn in (home, open_gripper, abort_safely):
            fn(self._driver, shim)
        elif fn is retry_grasp:
            fn(self._driver, shim, self._driver.read_tcp_pose())
        else:
            raise NotImplementedError(f"recovery primitive {primitive!r} requires extra wiring")
        self._emit({
            "ts": time.time(), "type": "recovery_invoked",
            "data": {"node_id": node_id, "primitive": primitive},
        })

    def _emit_safety(self, ev: SafetyEvent, node_id: str | None) -> None:
        self._emit({
            "ts": time.time(), "type": "safety_event",
            "data": {"node_id": node_id, "kind": ev.kind, "detail": ev.message},
        })

    def _emit(self, event: dict[str, Any]) -> None:
        if self._event_sink is not None:
            self._event_sink(event)


_GRIPPER_CODES = {"open": 0.0, "closed": 1.0, "grasping": 2.0}


def _encode_gripper(state: object) -> float:
    return _GRIPPER_CODES.get(str(state), -1.0)


class _SafetyShim:
    home_pose: Pose = (0.0, -668.0, 396.0, 3.14159, 0.0, 0.0)
    velocity_cap: float = 100.0
    watchdog_armed: bool = True

    def clamp_to_envelope(self, pose: Pose) -> Pose:
        return pose

    def disarm_watchdog(self) -> None:
        self.watchdog_armed = False


__all__ = [
    "ExecResult", "Plan", "PlanExecutor", "PlanNode",
    "Planner", "RunLoop", "RunSummary",
]
