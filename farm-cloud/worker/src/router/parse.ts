// Parse an LLM router response into a PlanDAG.
//
// Returns a discriminated union instead of throwing so callers can render a
// structured error to the user without try/catch boilerplate. Malformed JSON,
// missing fields, and dangling depends_on edges are all reported the same way.

export interface PlanNode {
  id: string;
  instruction: string;
  chosen_backend: string;
  reason: string;
  fallback_chain: string[];
  depends_on: string[];
}

export interface PlanDAG {
  plan_id: string;
  nodes: PlanNode[];
  max_attempts_per_node: number;
  max_wall_clock_ms: number;
}

export const DEFAULT_MAX_ATTEMPTS_PER_NODE = 2;
export const DEFAULT_MAX_WALL_CLOCK_MS = 5 * 60 * 1000;

export type ParseResult =
  | { ok: true; plan: PlanDAG; reasoning: string }
  | { ok: false; error: string; details?: unknown };

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === "string");
}

function parseNode(raw: unknown, index: number): PlanNode | string {
  if (!isObject(raw)) return `nodes[${index}] is not an object`;
  const { id, instruction, chosen_backend, reason, depends_on } = raw;
  if (typeof id !== "string" || id.length === 0)
    return `nodes[${index}].id missing or not a string`;
  if (typeof instruction !== "string")
    return `nodes[${index}].instruction missing or not a string`;
  if (typeof chosen_backend !== "string" || chosen_backend.length === 0)
    return `nodes[${index}].chosen_backend missing or not a string`;
  if (depends_on !== undefined && !isStringArray(depends_on))
    return `nodes[${index}].depends_on must be string[] when present`;
  return {
    id,
    instruction,
    chosen_backend,
    reason: typeof reason === "string" ? reason : "",
    // Filled in by fallback_chain.build at a later stage; LLM is not asked
    // to emit this field directly.
    fallback_chain: [],
    depends_on: depends_on ?? [],
  };
}

export function fromLLMResponse(raw: unknown): ParseResult {
  let data: unknown = raw;
  if (typeof raw === "string") {
    try {
      data = JSON.parse(raw);
    } catch (e) {
      return {
        ok: false,
        error: "response is not valid JSON",
        details: e instanceof Error ? e.message : String(e),
      };
    }
  }
  if (!isObject(data)) {
    return { ok: false, error: "response is not a JSON object" };
  }
  const { plan_id, nodes, reasoning } = data;
  if (typeof plan_id !== "string" || plan_id.length === 0) {
    return { ok: false, error: "plan_id missing or not a string" };
  }
  if (!Array.isArray(nodes) || nodes.length === 0) {
    return { ok: false, error: "nodes must be a non-empty array" };
  }
  const parsedNodes: PlanNode[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = parseNode(nodes[i], i);
    if (typeof node === "string") return { ok: false, error: node };
    parsedNodes.push(node);
  }
  const ids = new Set(parsedNodes.map((n) => n.id));
  if (ids.size !== parsedNodes.length) {
    return { ok: false, error: "node ids are not unique" };
  }
  for (const n of parsedNodes) {
    for (const dep of n.depends_on) {
      if (!ids.has(dep)) {
        return {
          ok: false,
          error: `node ${n.id} depends_on unknown node ${dep}`,
        };
      }
    }
  }
  return {
    ok: true,
    plan: {
      plan_id,
      nodes: parsedNodes,
      max_attempts_per_node: DEFAULT_MAX_ATTEMPTS_PER_NODE,
      max_wall_clock_ms: DEFAULT_MAX_WALL_CLOCK_MS,
    },
    reasoning: typeof reasoning === "string" ? reasoning : "",
  };
}
