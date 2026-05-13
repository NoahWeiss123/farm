// Build the LLM router prompt from a task string + a list of capability cards.
//
// Output is deterministic for a given input so prompt caching at the AI Gateway
// layer can hit on repeated calls. Cards are serialized in the order supplied;
// the caller is responsible for ordering if a canonical form is desired.

export type BackendRole = "planner" | "controller" | "critic";
export type Determinism = "deterministic" | "stochastic" | "seeded";

export interface SkillEntry {
  name: string;
  confidence: number;
  learned_from: number;
}

export interface CapabilityCard {
  id: string;
  name: string;
  roles: BackendRole[];
  embodiment?: {
    arm?: string;
    dof?: number;
    action_space?: string;
    control_rate_hz?: number;
  };
  input_modalities?: string[];
  camera_views?: string[];
  skills: SkillEntry[];
  latency?: {
    p50_ms_per_chunk?: number;
    p99_ms_per_chunk?: number;
  };
  cost_per_chunk_usd?: number;
  determinism?: Determinism;
  safety?: {
    requires_envelope?: boolean;
    supports_velocity_cap?: boolean;
  };
  fallbacks?: string[];
}

export interface Constraints {
  max_chunks?: number;
  max_cost_usd?: number;
}

const SYSTEM = [
  "You are the FARM router. You decompose a robotics task into a plan DAG and",
  "assign each node to a backend chosen from the supplied capability cards.",
  "Respond with one JSON object only, no prose, matching this shape:",
  '{"plan_id": string, "reasoning": string, "nodes":',
  '  [{"id": string, "instruction": string, "chosen_backend": string,',
  '    "reason": string, "depends_on": [string]}]}',
  "Rules:",
  "- chosen_backend must be a card id from CARDS, or 'classical-planner' when",
  "  no card has the relevant skill.",
  "- depends_on lists earlier node ids; the first node has [].",
  "- Keep the plan minimal; one node per discrete subtask.",
  "- Prefer higher skill confidence; tie-break on lower cost_per_chunk_usd.",
].join("\n");

function serializeCard(card: CapabilityCard): string {
  const skills = card.skills
    .map((s) => `    ${s.name}: confidence=${s.confidence} demos=${s.learned_from}`)
    .join("\n");
  const fallbacks = card.fallbacks?.length
    ? card.fallbacks.join(", ")
    : "(none)";
  const lat = card.latency
    ? `p50=${card.latency.p50_ms_per_chunk ?? "?"}ms p99=${card.latency.p99_ms_per_chunk ?? "?"}ms`
    : "(unknown)";
  return [
    `- id: ${card.id}`,
    `  name: ${card.name}`,
    `  roles: ${card.roles.join(",")}`,
    `  determinism: ${card.determinism ?? "unspecified"}`,
    `  cost_per_chunk_usd: ${card.cost_per_chunk_usd ?? "?"}`,
    `  latency: ${lat}`,
    `  skills:\n${skills || "    (none)"}`,
    `  fallbacks: ${fallbacks}`,
  ].join("\n");
}

function serializeConstraints(c: Constraints | undefined): string {
  if (!c) return "(none)";
  const parts: string[] = [];
  if (typeof c.max_chunks === "number") parts.push(`max_chunks=${c.max_chunks}`);
  if (typeof c.max_cost_usd === "number")
    parts.push(`max_cost_usd=${c.max_cost_usd}`);
  return parts.length ? parts.join(" ") : "(none)";
}

export function build(
  task: string,
  cards: CapabilityCard[],
  constraints?: Constraints,
): string {
  const body = cards.length
    ? cards.map(serializeCard).join("\n")
    : "(no capability cards supplied)";
  return [
    SYSTEM,
    "",
    "CARDS:",
    body,
    "",
    `CONSTRAINTS: ${serializeConstraints(constraints)}`,
    "",
    `TASK: ${task}`,
  ].join("\n");
}
