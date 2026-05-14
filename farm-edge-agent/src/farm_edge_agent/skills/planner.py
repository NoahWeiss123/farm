"""Stub planner — turns a task string into a single-node Plan.

Replaced in Phase 9 by the hierarchical GPT decomposer (skill library lookup
+ recursive sub-task decomposition). For Phase 3 we just shuttle the task
string through as one node so the RunLoop can exercise its safety,
recording, and executor wiring without an LLM dependency.
"""

from __future__ import annotations

import hashlib

from farm_edge_agent.run_loop import Plan, PlanNode


class StubPlanner:
    def plan(self, task: str, *, run_id: str) -> Plan:
        h = hashlib.sha1(task.encode("utf-8")).hexdigest()[:8]
        return Plan(
            plan_id=f"plan_stub_{h}",
            nodes=[PlanNode(id="n1", instruction=task, backend="sim")],
            reasoning="stub planner: single-node plan",
        )


__all__ = ["StubPlanner"]
