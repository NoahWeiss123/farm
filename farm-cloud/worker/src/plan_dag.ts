// PlanDAG: the structured plan the router LLM emits. Each node has a primary
// backend, an ordered fallback chain, and explicit edge dependencies.

export type RecoveryPrimitive =
  | "home"
  | "open_gripper"
  | "relocalize"
  | "abort_safely";

export interface PlanNode {
  id: string;
  instruction: string;
  chosen_backend: string;
  fallback_chain: string[];
  depends_on: string[];
  // Recovery primitive to invoke before any fallback backend takes over.
  // Defaults applied by `runtimeChainFor` when unset.
  recovery_primitive?: RecoveryPrimitive;
}

export interface PlanDAG {
  plan_id: string;
  nodes: PlanNode[];
  max_attempts_per_node: number;
  // Wall-clock budget for the whole run, in milliseconds. DESIGN.md default is
  // 5 minutes.
  max_wall_clock_ms: number;
}

export const DEFAULT_MAX_ATTEMPTS_PER_NODE = 2;
export const DEFAULT_MAX_WALL_CLOCK_MS = 5 * 60 * 1000;

export class PlanError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PlanError";
  }
}

export function validatePlan(plan: PlanDAG): void {
  if (plan.nodes.length === 0) {
    throw new PlanError("plan has no nodes");
  }
  if (plan.max_attempts_per_node < 1) {
    throw new PlanError("max_attempts_per_node must be >= 1");
  }
  if (plan.max_wall_clock_ms <= 0) {
    throw new PlanError("max_wall_clock_ms must be > 0");
  }
  const ids = new Set<string>();
  for (const node of plan.nodes) {
    if (ids.has(node.id)) {
      throw new PlanError(`duplicate node id: ${node.id}`);
    }
    ids.add(node.id);
  }
  for (const node of plan.nodes) {
    for (const dep of node.depends_on) {
      if (!ids.has(dep)) {
        throw new PlanError(
          `node ${node.id} depends on unknown node ${dep}`,
        );
      }
    }
  }
}

// Returns a topologically-ordered execution sequence. Throws on cycles.
export function walkOrder(plan: PlanDAG): PlanNode[] {
  const byId = new Map(plan.nodes.map((n) => [n.id, n]));
  const indegree = new Map<string, number>();
  for (const node of plan.nodes) {
    indegree.set(node.id, node.depends_on.length);
  }
  const out: PlanNode[] = [];
  const ready: string[] = [];
  for (const [id, deg] of indegree) {
    if (deg === 0) ready.push(id);
  }
  // Preserve declared order among ready nodes for determinism.
  ready.sort((a, b) => declaredIndex(plan, a) - declaredIndex(plan, b));
  while (ready.length > 0) {
    const id = ready.shift()!;
    const node = byId.get(id)!;
    out.push(node);
    for (const other of plan.nodes) {
      if (other.depends_on.includes(id)) {
        const deg = (indegree.get(other.id) ?? 0) - 1;
        indegree.set(other.id, deg);
        if (deg === 0) ready.push(other.id);
      }
    }
    ready.sort((a, b) => declaredIndex(plan, a) - declaredIndex(plan, b));
  }
  if (out.length !== plan.nodes.length) {
    throw new PlanError("plan contains a cycle");
  }
  return out;
}

function declaredIndex(plan: PlanDAG, id: string): number {
  return plan.nodes.findIndex((n) => n.id === id);
}

// The runtime chain a node will try: primary first, then declared fallbacks.
// Duplicate ids are removed while preserving first-occurrence order.
export function runtimeChainFor(node: PlanNode): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const id of [node.chosen_backend, ...node.fallback_chain]) {
    if (!seen.has(id)) {
      seen.add(id);
      out.push(id);
    }
  }
  return out;
}
