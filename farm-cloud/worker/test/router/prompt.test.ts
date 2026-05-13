import { describe, expect, it } from "vitest";
import { build, type CapabilityCard } from "../../src/router/prompt";

const PI05: CapabilityCard = {
  id: "pi05-ufactory-ft-v1",
  name: "pi0.5 fine-tuned",
  roles: ["controller"],
  skills: [
    { name: "pick", confidence: 0.8, learned_from: 240 },
    { name: "place", confidence: 0.85, learned_from: 240 },
  ],
  latency: { p50_ms_per_chunk: 95, p99_ms_per_chunk: 220 },
  cost_per_chunk_usd: 0.0008,
  determinism: "stochastic",
  fallbacks: ["classical-planner"],
};

const CLASSICAL: CapabilityCard = {
  id: "classical-planner",
  name: "classical motion planner",
  roles: ["controller"],
  skills: [
    { name: "pick", confidence: 0.5, learned_from: 0 },
    { name: "place", confidence: 0.5, learned_from: 0 },
  ],
  determinism: "deterministic",
  cost_per_chunk_usd: 0,
};

describe("prompt.build", () => {
  it("is deterministic for the same input (golden)", () => {
    const a = build("pick the red block", [PI05, CLASSICAL]);
    const b = build("pick the red block", [PI05, CLASSICAL]);
    expect(a).toBe(b);
  });

  it("matches the locked golden string", () => {
    const out = build("pick the red block", [PI05, CLASSICAL]);
    expect(out).toMatchInlineSnapshot(`
      "You are the FARM router. You decompose a robotics task into a plan DAG and
      assign each node to a backend chosen from the supplied capability cards.
      Respond with one JSON object only, no prose, matching this shape:
      {"plan_id": string, "reasoning": string, "nodes":
        [{"id": string, "instruction": string, "chosen_backend": string,
          "reason": string, "depends_on": [string]}]}
      Rules:
      - chosen_backend must be a card id from CARDS, or 'classical-planner' when
        no card has the relevant skill.
      - depends_on lists earlier node ids; the first node has [].
      - Keep the plan minimal; one node per discrete subtask.
      - Prefer higher skill confidence; tie-break on lower cost_per_chunk_usd.

      CARDS:
      - id: pi05-ufactory-ft-v1
        name: pi0.5 fine-tuned
        roles: controller
        determinism: stochastic
        cost_per_chunk_usd: 0.0008
        latency: p50=95ms p99=220ms
        skills:
          pick: confidence=0.8 demos=240
          place: confidence=0.85 demos=240
        fallbacks: classical-planner
      - id: classical-planner
        name: classical motion planner
        roles: controller
        determinism: deterministic
        cost_per_chunk_usd: 0
        latency: (unknown)
        skills:
          pick: confidence=0.5 demos=0
          place: confidence=0.5 demos=0
        fallbacks: (none)

      CONSTRAINTS: (none)

      TASK: pick the red block"
    `);
  });

  it("includes the task verbatim", () => {
    const out = build("stack the green cube on the red cube", [PI05]);
    expect(out).toContain("TASK: stack the green cube on the red cube");
  });

  it("renders supplied constraints", () => {
    const out = build("pick", [PI05], { max_chunks: 12, max_cost_usd: 0.05 });
    expect(out).toContain("CONSTRAINTS: max_chunks=12 max_cost_usd=0.05");
  });

  it("emits a placeholder when no cards are supplied", () => {
    const out = build("idle", []);
    expect(out).toContain("(no capability cards supplied)");
  });
});
