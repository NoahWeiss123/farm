// POST /v1/plans handler. Builds a router prompt from capability cards, calls
// Claude Sonnet 4.6 via the AI Gateway binding (falling back to a direct
// Anthropic call when the binding is missing), parses the response into a
// PlanDAG, and decorates each node with a default fallback chain.
//
// The LLM call is funnelled through `routeWithLLM`, a single interface the
// test suite swaps for a fixture so no real network calls happen in CI.

import { Hono } from "hono";
import {
  build as buildPrompt,
  type CapabilityCard,
  type Constraints,
} from "./router/prompt";
import {
  fromLLMResponse,
  type PlanDAG,
  type PlanNode,
} from "./router/parse";
import {
  build as buildFallbackChain,
  CLASSICAL_BACKEND_ID,
} from "./router/fallback_chain";

export const ROUTER_MODEL = "claude-sonnet-4-6";

export interface PlannerEnv {
  ANTHROPIC_API_KEY?: string;
  "ai-gateway"?: Fetcher;
}

export interface RouterResponse {
  raw: unknown;
}

export type LLMRouter = (
  prompt: string,
  env: PlannerEnv,
) => Promise<RouterResponse>;

export interface PlanRequestBody {
  task: string;
  capability_cards: CapabilityCard[];
  constraints?: Constraints;
}

export interface PlanResponseBody {
  plan: PlanDAG;
  reasoning: string;
}

export interface PlanErrorBody {
  error: string;
  details?: unknown;
}

function validateBody(body: unknown): PlanRequestBody | string {
  if (typeof body !== "object" || body === null) return "body must be a JSON object";
  const b = body as Record<string, unknown>;
  if (typeof b.task !== "string" || b.task.length === 0)
    return "task must be a non-empty string";
  if (!Array.isArray(b.capability_cards))
    return "capability_cards must be an array";
  return {
    task: b.task,
    capability_cards: b.capability_cards as CapabilityCard[],
    constraints: (b.constraints as Constraints | undefined) ?? undefined,
  };
}

function hasSkill(card: CapabilityCard, skill: string): boolean {
  return card.skills.some((s) => s.name === skill);
}

function inferRequiredSkills(task: string, cards: CapabilityCard[]): string[] {
  const lower = task.toLowerCase();
  const skills = new Set<string>();
  for (const c of cards) {
    for (const s of c.skills) {
      if (lower.includes(s.name)) skills.add(s.name);
    }
  }
  return [...skills];
}

function syntheticClassicalPlan(
  task: string,
  cards: CapabilityCard[],
): { plan: PlanDAG; reasoning: string } {
  const required = inferRequiredSkills(task, cards);
  const matchable = required.length === 0
    ? false
    : required.some((skill) => cards.some((c) => hasSkill(c, skill)));
  const reason = matchable
    ? "router fell through to classical"
    : "no backend matched skill";
  return {
    plan: {
      plan_id: `plan_classical_${task.length}`,
      nodes: [
        {
          id: "n1",
          instruction: task,
          chosen_backend: CLASSICAL_BACKEND_ID,
          reason,
          fallback_chain: [],
          depends_on: [],
        },
      ],
      max_attempts_per_node: 2,
      max_wall_clock_ms: 5 * 60 * 1000,
    },
    reasoning: reason,
  };
}

function decorateFallbacks(plan: PlanDAG, cards: CapabilityCard[]): PlanDAG {
  const byId = new Map(cards.map((c) => [c.id, c]));
  const decorated: PlanNode[] = plan.nodes.map((node) => {
    if (node.chosen_backend === CLASSICAL_BACKEND_ID) {
      return { ...node, fallback_chain: [] };
    }
    const primary = byId.get(node.chosen_backend);
    if (!primary) {
      // Router named a backend we don't have a card for. Fall through to
      // classical on the first attempt.
      return { ...node, fallback_chain: [CLASSICAL_BACKEND_ID] };
    }
    return { ...node, fallback_chain: buildFallbackChain(primary, cards) };
  });
  return { ...plan, nodes: decorated };
}

async function defaultRouter(
  prompt: string,
  env: PlannerEnv,
): Promise<RouterResponse> {
  const payload = {
    model: ROUTER_MODEL,
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  };
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
  };
  if (env.ANTHROPIC_API_KEY) headers["x-api-key"] = env.ANTHROPIC_API_KEY;
  const gateway = env["ai-gateway"];
  const url = gateway
    ? "https://gateway/anthropic/v1/messages"
    : "https://api.anthropic.com/v1/messages";
  const init: RequestInit = {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  };
  const res = gateway
    ? await gateway.fetch(url, init)
    : await globalThis.fetch(url, init);
  if (!res.ok) {
    throw new Error(`router LLM call failed: ${res.status}`);
  }
  const data = (await res.json()) as {
    content?: { text?: string }[];
  };
  const text = data.content?.[0]?.text ?? "";
  return { raw: text };
}

export interface HandlerOptions {
  router?: LLMRouter;
}

export async function handlePlan(
  body: unknown,
  env: PlannerEnv,
  opts: HandlerOptions = {},
): Promise<{ status: number; body: PlanResponseBody | PlanErrorBody }> {
  const parsed = validateBody(body);
  if (typeof parsed === "string") {
    return { status: 400, body: { error: parsed } };
  }
  const prompt = buildPrompt(
    parsed.task,
    parsed.capability_cards,
    parsed.constraints,
  );
  const router = opts.router ?? defaultRouter;
  let raw: unknown;
  try {
    raw = (await router(prompt, env)).raw;
  } catch (e) {
    return {
      status: 502,
      body: {
        error: "router LLM call failed",
        details: e instanceof Error ? e.message : String(e),
      },
    };
  }
  const result = fromLLMResponse(raw);
  if (!result.ok) {
    const fallback = syntheticClassicalPlan(
      parsed.task,
      parsed.capability_cards,
    );
    return {
      status: 200,
      body: {
        plan: decorateFallbacks(fallback.plan, parsed.capability_cards),
        reasoning: `${result.error}; ${fallback.reasoning}`,
      },
    };
  }
  return {
    status: 200,
    body: {
      plan: decorateFallbacks(result.plan, parsed.capability_cards),
      reasoning: result.reasoning,
    },
  };
}

export function plannerApp(opts: HandlerOptions = {}): Hono<{
  Bindings: PlannerEnv;
}> {
  const app = new Hono<{ Bindings: PlannerEnv }>();
  app.post("/v1/plans", async (c) => {
    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      return c.json({ error: "body is not valid JSON" }, 400);
    }
    const { status, body: out } = await handlePlan(body, c.env, opts);
    return c.json(out, status as 200 | 400 | 502);
  });
  return app;
}
