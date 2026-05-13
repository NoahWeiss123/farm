import { describe, expect, it } from "vitest";
import {
  build,
  CLASSICAL_BACKEND_ID,
} from "../../src/router/fallback_chain";
import type { CapabilityCard } from "../../src/router/prompt";

function card(over: Partial<CapabilityCard>): CapabilityCard {
  return {
    id: "stub",
    name: "stub",
    roles: ["controller"],
    skills: [],
    ...over,
  };
}

const PI05_FT = card({
  id: "pi05-ft",
  name: "pi0.5 fine-tuned",
  skills: [
    { name: "pick", confidence: 0.85, learned_from: 240 },
    { name: "place", confidence: 0.8, learned_from: 240 },
  ],
  cost_per_chunk_usd: 0.0008,
});

const PI05_BASE = card({
  id: "pi05-base",
  name: "pi0.5 base",
  skills: [
    { name: "pick", confidence: 0.6, learned_from: 0 },
    { name: "place", confidence: 0.6, learned_from: 0 },
  ],
  cost_per_chunk_usd: 0.0005,
});

const GEMINI = card({
  id: "gemini-robotics",
  name: "gemini robotics",
  skills: [
    { name: "pick", confidence: 0.7, learned_from: 0 },
    { name: "pour", confidence: 0.7, learned_from: 0 },
  ],
  cost_per_chunk_usd: 0.002,
});

const CLASSICAL = card({
  id: CLASSICAL_BACKEND_ID,
  name: "classical motion planner",
  skills: [{ name: "pick", confidence: 0.5, learned_from: 0 }],
});

const PLANNER_ONLY = card({
  id: "claude-planner",
  name: "claude as planner",
  roles: ["planner"],
  skills: [{ name: "pick", confidence: 0.9, learned_from: 0 }],
});

describe("fallback_chain.build", () => {
  it("always ends with classical-planner", () => {
    const chain = build(PI05_FT, [PI05_FT, PI05_BASE, GEMINI]);
    expect(chain[chain.length - 1]).toBe(CLASSICAL_BACKEND_ID);
  });

  it("appends classical-planner even when no other card overlaps", () => {
    const lonely = card({
      id: "lonely",
      skills: [{ name: "wave", confidence: 0.4, learned_from: 0 }],
    });
    const chain = build(lonely, [lonely]);
    expect(chain).toEqual([CLASSICAL_BACKEND_ID]);
  });

  it("orders overlapping controllers by descending skill confidence", () => {
    const chain = build(PI05_FT, [PI05_FT, PI05_BASE, GEMINI, CLASSICAL]);
    expect(chain).toEqual(["gemini-robotics", "pi05-base", CLASSICAL_BACKEND_ID]);
  });

  it("breaks ties on cost_per_chunk_usd ascending", () => {
    const cheap = card({
      id: "cheap",
      skills: [{ name: "pick", confidence: 0.7, learned_from: 0 }],
      cost_per_chunk_usd: 0.0001,
    });
    const pricey = card({
      id: "pricey",
      skills: [{ name: "pick", confidence: 0.7, learned_from: 0 }],
      cost_per_chunk_usd: 0.01,
    });
    const chain = build(PI05_FT, [PI05_FT, cheap, pricey]);
    expect(chain).toEqual(["cheap", "pricey", CLASSICAL_BACKEND_ID]);
  });

  it("excludes the primary card from its own fallback chain", () => {
    const chain = build(PI05_FT, [PI05_FT, PI05_BASE]);
    expect(chain).not.toContain(PI05_FT.id);
  });

  it("excludes cards that share no skills with the primary", () => {
    const unrelated = card({
      id: "unrelated",
      skills: [{ name: "wipe", confidence: 0.9, learned_from: 10 }],
    });
    const chain = build(PI05_FT, [PI05_FT, unrelated]);
    expect(chain).not.toContain("unrelated");
  });

  it("ignores classical when it appears in allCards (always added at the end)", () => {
    const chain = build(PI05_FT, [PI05_FT, CLASSICAL]);
    expect(chain.filter((id) => id === CLASSICAL_BACKEND_ID)).toHaveLength(1);
  });

  it("ignores planner-role cards when building a controller fallback chain", () => {
    const chain = build(PI05_FT, [PI05_FT, PLANNER_ONLY]);
    expect(chain).not.toContain(PLANNER_ONLY.id);
    expect(chain).toEqual([CLASSICAL_BACKEND_ID]);
  });
});
