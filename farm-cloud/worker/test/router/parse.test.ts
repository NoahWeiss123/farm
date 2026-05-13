import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAX_ATTEMPTS_PER_NODE,
  DEFAULT_MAX_WALL_CLOCK_MS,
  fromLLMResponse,
} from "../../src/router/parse";

const validResponse = {
  plan_id: "plan_abc",
  reasoning: "single pick step",
  nodes: [
    {
      id: "n1",
      instruction: "pick the red block",
      chosen_backend: "pi05-ufactory-ft-v1",
      reason: "highest skill confidence for pick",
      depends_on: [],
    },
  ],
};

describe("parse.fromLLMResponse", () => {
  it("accepts a well-formed JSON object", () => {
    const r = fromLLMResponse(validResponse);
    if (!r.ok) throw new Error(r.error);
    expect(r.plan.plan_id).toBe("plan_abc");
    expect(r.plan.nodes).toHaveLength(1);
    expect(r.plan.nodes[0]!.fallback_chain).toEqual([]);
    expect(r.plan.max_attempts_per_node).toBe(DEFAULT_MAX_ATTEMPTS_PER_NODE);
    expect(r.plan.max_wall_clock_ms).toBe(DEFAULT_MAX_WALL_CLOCK_MS);
    expect(r.reasoning).toBe("single pick step");
  });

  it("accepts a JSON string and parses it", () => {
    const r = fromLLMResponse(JSON.stringify(validResponse));
    expect(r.ok).toBe(true);
  });

  it("returns a structured error for malformed JSON, not a throw", () => {
    const r = fromLLMResponse("{not json");
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("not valid JSON");
  });

  it("rejects a non-object top-level value", () => {
    const r = fromLLMResponse("[1, 2, 3]");
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("not a JSON object");
  });

  it("rejects missing plan_id", () => {
    const r = fromLLMResponse({ nodes: validResponse.nodes });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("plan_id");
  });

  it("rejects an empty nodes array", () => {
    const r = fromLLMResponse({ plan_id: "p", nodes: [] });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("non-empty");
  });

  it("rejects a node missing chosen_backend", () => {
    const r = fromLLMResponse({
      plan_id: "p",
      nodes: [{ id: "n1", instruction: "do it" }],
    });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("chosen_backend");
  });

  it("rejects duplicate node ids", () => {
    const r = fromLLMResponse({
      plan_id: "p",
      nodes: [
        { id: "n1", instruction: "a", chosen_backend: "x", depends_on: [] },
        { id: "n1", instruction: "b", chosen_backend: "x", depends_on: [] },
      ],
    });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("not unique");
  });

  it("rejects depends_on referring to an unknown node", () => {
    const r = fromLLMResponse({
      plan_id: "p",
      nodes: [
        {
          id: "n1",
          instruction: "a",
          chosen_backend: "x",
          depends_on: ["ghost"],
        },
      ],
    });
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.error).toContain("ghost");
  });

  it("leaves fallback_chain empty for downstream backfill", () => {
    const r = fromLLMResponse(validResponse);
    if (!r.ok) throw new Error(r.error);
    for (const node of r.plan.nodes) {
      expect(node.fallback_chain).toEqual([]);
    }
  });
});
