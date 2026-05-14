import { describe, expect, it } from "vitest";
import {
  handlePlan,
  plannerApp,
  type LLMRouter,
  type PlanRequestBody,
  type PlanResponseBody,
  type PlanErrorBody,
} from "../src/planner";
import type { CapabilityCard } from "../src/router/prompt";
import { CLASSICAL_BACKEND_ID } from "../src/router/fallback_chain";

const PI05: CapabilityCard = {
  id: "pi05-ft",
  name: "pi0.5 fine-tuned",
  roles: ["controller"],
  skills: [
    { name: "pick", confidence: 0.85, learned_from: 240 },
    { name: "place", confidence: 0.8, learned_from: 240 },
  ],
  cost_per_chunk_usd: 0.0008,
};

const GEMINI: CapabilityCard = {
  id: "gemini-robotics",
  name: "gemini robotics",
  roles: ["controller"],
  skills: [
    { name: "pick", confidence: 0.7, learned_from: 0 },
    { name: "pour", confidence: 0.7, learned_from: 0 },
  ],
  cost_per_chunk_usd: 0.002,
};

function fixtureRouter(out: unknown): LLMRouter {
  return async () => ({ raw: out });
}

function pickPlan(backend = "pi05-ft"): unknown {
  return {
    plan_id: "plan_pick_1",
    reasoning: "single pick step",
    nodes: [
      {
        id: "n1",
        instruction: "pick the red block",
        chosen_backend: backend,
        reason: "highest skill confidence",
        depends_on: [],
      },
    ],
  };
}

describe("handlePlan", () => {
  it("returns a 200 with a decorated plan when the LLM emits a valid plan", async () => {
    const body: PlanRequestBody = {
      task: "pick the red block",
      capability_cards: [PI05, GEMINI],
    };
    const res = await handlePlan(body, {}, { router: fixtureRouter(pickPlan()) });
    expect(res.status).toBe(200);
    const out = res.body as PlanResponseBody;
    expect(out.plan.nodes).toHaveLength(1);
    expect(out.plan.nodes[0]!.chosen_backend).toBe("pi05-ft");
    expect(out.plan.nodes[0]!.fallback_chain).toEqual([
      "gemini-robotics",
      CLASSICAL_BACKEND_ID,
    ]);
    expect(out.reasoning).toBe("single pick step");
  });

  it("rejects a body missing the task field", async () => {
    const res = await handlePlan({ capability_cards: [] }, {}, {
      router: fixtureRouter(pickPlan()),
    });
    expect(res.status).toBe(400);
    expect((res.body as PlanErrorBody).error).toContain("task");
  });

  it("rejects a non-object body", async () => {
    const res = await handlePlan("not a body", {}, {
      router: fixtureRouter(pickPlan()),
    });
    expect(res.status).toBe(400);
  });

  it("returns 502 when the router throws", async () => {
    const router: LLMRouter = async () => {
      throw new Error("openai 503");
    };
    const res = await handlePlan(
      { task: "pick", capability_cards: [PI05] },
      {},
      { router },
    );
    expect(res.status).toBe(502);
    expect((res.body as PlanErrorBody).error).toContain("router");
  });

  it("falls back to classical when no card has the required skill", async () => {
    const body: PlanRequestBody = {
      task: "pour the water carefully",
      capability_cards: [PI05],
    };
    // LLM emits something the parser can't accept; handler synthesizes a
    // classical plan and reports "no backend matched skill".
    const res = await handlePlan(body, {}, {
      router: fixtureRouter({ not: "a plan" }),
    });
    expect(res.status).toBe(200);
    const out = res.body as PlanResponseBody;
    expect(out.plan.nodes[0]!.chosen_backend).toBe(CLASSICAL_BACKEND_ID);
    expect(out.reasoning).toContain("no backend matched skill");
  });

  it("falls back to classical on parser failure even when skills do match", async () => {
    const body: PlanRequestBody = {
      task: "pick the red block",
      capability_cards: [PI05],
    };
    const res = await handlePlan(body, {}, {
      router: fixtureRouter("not even json {"),
    });
    expect(res.status).toBe(200);
    const out = res.body as PlanResponseBody;
    expect(out.plan.nodes[0]!.chosen_backend).toBe(CLASSICAL_BACKEND_ID);
    expect(out.reasoning).toContain("router fell through to classical");
  });

  it("attaches [classical] as the fallback when the LLM names an unknown backend", async () => {
    const body: PlanRequestBody = {
      task: "pick",
      capability_cards: [PI05],
    };
    const res = await handlePlan(body, {}, {
      router: fixtureRouter(pickPlan("ghost-backend")),
    });
    const out = res.body as PlanResponseBody;
    expect(out.plan.nodes[0]!.chosen_backend).toBe("ghost-backend");
    expect(out.plan.nodes[0]!.fallback_chain).toEqual([CLASSICAL_BACKEND_ID]);
  });

  it("does not append fallbacks when the LLM picks classical directly", async () => {
    const body: PlanRequestBody = {
      task: "pick",
      capability_cards: [PI05],
    };
    const res = await handlePlan(body, {}, {
      router: fixtureRouter(pickPlan(CLASSICAL_BACKEND_ID)),
    });
    const out = res.body as PlanResponseBody;
    expect(out.plan.nodes[0]!.chosen_backend).toBe(CLASSICAL_BACKEND_ID);
    expect(out.plan.nodes[0]!.fallback_chain).toEqual([]);
  });
});

describe("default OpenAI router", () => {
  it("uses fetch with bearer auth and chat completions URL when no gateway is bound", async () => {
    const calls: { url: string; init: RequestInit }[] = [];
    const fakeFetch = (async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return new Response(
        JSON.stringify({
          choices: [{ message: { content: JSON.stringify({ plan_id: "p", nodes: [] }) } }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = fakeFetch;
    try {
      const { defaultRouterForTests } = await import("../src/planner");
      const out = await defaultRouterForTests("hello", { OPENAI_API_KEY: "sk-test" });
      expect(typeof out.raw).toBe("string");
      expect(calls).toHaveLength(1);
      expect(calls[0]!.url).toBe("https://api.openai.com/v1/chat/completions");
      const headers = calls[0]!.init.headers as Record<string, string>;
      expect(headers.authorization).toBe("Bearer sk-test");
      const body = JSON.parse(calls[0]!.init.body as string);
      expect(body.model).toBe("gpt-4o");
      expect(body.response_format).toEqual({ type: "json_object" });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("throws when no api key is configured", async () => {
    const { defaultRouterForTests } = await import("../src/planner");
    await expect(defaultRouterForTests("p", {})).rejects.toThrow(/OPENAI_API_KEY/);
  });

  it("honors OPENAI_MODEL override", async () => {
    const calls: { init: RequestInit }[] = [];
    const fakeFetch = (async (_url: string, init: RequestInit) => {
      calls.push({ init });
      return new Response(JSON.stringify({ choices: [{ message: { content: "{}" } }] }), {
        status: 200,
      });
    }) as unknown as typeof fetch;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = fakeFetch;
    try {
      const { defaultRouterForTests } = await import("../src/planner");
      await defaultRouterForTests("p", {
        OPENAI_API_KEY: "sk-test",
        OPENAI_MODEL: "gpt-5",
      });
      const body = JSON.parse(calls[0]!.init.body as string);
      expect(body.model).toBe("gpt-5");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe("plannerApp", () => {
  it("POST /v1/plans returns 200 with the decorated plan", async () => {
    const app = plannerApp({ router: fixtureRouter(pickPlan()) });
    const res = await app.request("/v1/plans", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        task: "pick the red block",
        capability_cards: [PI05, GEMINI],
      }),
    });
    expect(res.status).toBe(200);
    const out = (await res.json()) as PlanResponseBody;
    expect(out.plan.nodes[0]!.chosen_backend).toBe("pi05-ft");
  });

  it("POST /v1/plans returns 400 when the body is not valid JSON", async () => {
    const app = plannerApp({ router: fixtureRouter(pickPlan()) });
    const res = await app.request("/v1/plans", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "not json",
    });
    expect(res.status).toBe(400);
  });
});
