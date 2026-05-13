import { describe, expect, it } from "vitest";
import {
  dispatchPlan,
  type DispatchDeps,
  type RunEvent,
} from "../src/dispatcher";
import type { PlanDAG, PlanNode } from "../src/plan_dag";
import {
  type Backend,
  type BackendInput,
  type BackendOutcome,
  type RecoveryRunner,
} from "../src/fallback";
import { initialRunState, type ActionChunk, type RunState } from "../src/run_state";
import { ClassicalBackend } from "../src/backends/classical";

class StubBackend implements Backend {
  readonly calls: BackendInput[] = [];
  constructor(
    readonly id: string,
    private readonly script: BackendOutcome[],
  ) {}
  async run(input: BackendInput): Promise<BackendOutcome> {
    this.calls.push(input);
    const next = this.script.shift();
    if (next === undefined) {
      throw new Error(`StubBackend ${this.id} ran out of scripted outcomes`);
    }
    return next;
  }
}

function chunk(chunk_id: number): ActionChunk {
  return {
    chunk_id,
    suggested_dwell_ms: 100,
    actions: [
      { dx: 0, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0, gripper: null },
    ],
  };
}

function withSummary(state: RunState, s: string): RunState {
  return { ...state, critic_summary: s };
}

function makePlan(nodes: PlanNode[], opts: Partial<PlanDAG> = {}): PlanDAG {
  return {
    plan_id: opts.plan_id ?? "p1",
    nodes,
    max_attempts_per_node: opts.max_attempts_per_node ?? 2,
    max_wall_clock_ms: opts.max_wall_clock_ms ?? 60_000,
  };
}

const noRecovery: RecoveryRunner = async (_, s) => s;

function mutableClock(): { now: () => number; set: (v: number) => void } {
  const ref = { value: 0 };
  return {
    now: () => ref.value,
    set: (v: number) => {
      ref.value = v;
    },
  };
}

describe("dispatchPlan happy path", () => {
  it("completes a 1-node plan with the classical stub", async () => {
    const plan = makePlan([
      {
        id: "n0",
        instruction: "place the cube",
        chosen_backend: "classical-planner",
        fallback_chain: [],
        depends_on: [],
      },
    ]);
    const deps: DispatchDeps = {
      backends: new Map<string, Backend>([
        ["classical-planner", new ClassicalBackend()],
      ]),
      recovery: noRecovery,
      now: () => 0,
    };
    const result = await dispatchPlan(
      { run_id: "r-happy", plan },
      deps,
    );
    expect(result.outcome).toBe("success");
    expect(result.events[0]?.type).toBe("run_started");
    const last = result.events[result.events.length - 1];
    expect(last?.type).toBe("run_completed");
    if (last?.type !== "run_completed") throw new Error("unreachable");
    expect(last.outcome).toBe("success");
    expect(result.final_state.current_node_id).toBe("n0");
    expect(result.final_state.last_completed_chunk_index).toBeGreaterThanOrEqual(
      0,
    );
  });
});

describe("dispatchPlan with two sequential nodes", () => {
  it("runs them in declared order honoring depends_on", async () => {
    const state = initialRunState("r-seq");
    const order: string[] = [];
    const a = new StubBackend("a", [
      {
        kind: "success",
        chunks: [chunk(0)],
        final_state: withSummary(state, "a"),
      },
    ]);
    const b = new StubBackend("b", [
      {
        kind: "success",
        chunks: [chunk(1)],
        final_state: withSummary(state, "b"),
      },
    ]);
    const wrap = (id: string, inner: Backend): Backend => ({
      id,
      run: async (input) => {
        order.push(id);
        return inner.run(input);
      },
    });
    const plan = makePlan([
      {
        id: "n0",
        instruction: "first",
        chosen_backend: "a",
        fallback_chain: [],
        depends_on: [],
      },
      {
        id: "n1",
        instruction: "second",
        chosen_backend: "b",
        fallback_chain: [],
        depends_on: ["n0"],
      },
    ]);
    const result = await dispatchPlan(
      { run_id: "r-seq", plan },
      {
        backends: new Map<string, Backend>([
          ["a", wrap("a", a)],
          ["b", wrap("b", b)],
        ]),
        recovery: noRecovery,
        now: () => 0,
      },
    );
    expect(result.outcome).toBe("success");
    expect(order).toEqual(["a", "b"]);
    const nodeStarts = result.events.filter(
      (e): e is Extract<RunEvent, { type: "node_started" }> =>
        e.type === "node_started",
    );
    expect(nodeStarts.map((e) => e.node_id)).toEqual(["n0", "n1"]);
  });
});

describe("dispatchPlan fallback chain", () => {
  it("falls over to next backend with handed-off state", async () => {
    const state = initialRunState("r-fb");
    const primary = new StubBackend("primary", [
      { kind: "error", reason: "model_oom" },
      { kind: "error", reason: "model_oom" },
    ]);
    const secondary = new StubBackend("secondary", [
      {
        kind: "success",
        chunks: [chunk(0)],
        final_state: withSummary(state, "secondary ok"),
      },
    ]);
    const recoveryCalls: string[] = [];
    const recovery: RecoveryRunner = async (primitive, s) => {
      recoveryCalls.push(primitive);
      return withSummary(s, `recovered:${primitive}`);
    };
    const plan = makePlan([
      {
        id: "n0",
        instruction: "pick",
        chosen_backend: "primary",
        fallback_chain: ["secondary"],
        depends_on: [],
      },
    ]);
    const result = await dispatchPlan(
      { run_id: "r-fb", plan },
      {
        backends: new Map<string, Backend>([
          ["primary", primary],
          ["secondary", secondary],
        ]),
        recovery,
        now: () => 0,
      },
    );
    expect(result.outcome).toBe("success");
    expect(primary.calls).toHaveLength(2);
    expect(secondary.calls).toHaveLength(1);
    expect(recoveryCalls).toEqual(["home"]);
    expect(secondary.calls[0]?.state.critic_summary).toBe("recovered:home");
  });
});

describe("dispatchPlan timeout", () => {
  it("hard-stops with a timeout critic note when the budget is exhausted mid-run", async () => {
    const state = initialRunState("r-timeout");
    const clock = mutableClock();
    // Backend a finishes inside the budget. After it returns, we advance the
    // clock past the budget; the dispatcher must refuse to start node b and
    // emit a timeout critic note.
    const a: Backend = {
      id: "a",
      async run() {
        clock.set(40);
        return {
          kind: "success",
          chunks: [chunk(0)],
          final_state: withSummary(state, "a"),
        };
      },
    };
    const b = new StubBackend("b", [
      {
        kind: "success",
        chunks: [chunk(1)],
        final_state: withSummary(state, "b"),
      },
    ]);
    const plan = makePlan(
      [
        {
          id: "n0",
          instruction: "first",
          chosen_backend: "a",
          fallback_chain: [],
          depends_on: [],
        },
        {
          id: "n1",
          instruction: "second",
          chosen_backend: "b",
          fallback_chain: [],
          depends_on: ["n0"],
        },
      ],
      { max_wall_clock_ms: 30 },
    );
    const result = await dispatchPlan(
      { run_id: "r-timeout", plan },
      {
        backends: new Map<string, Backend>([
          ["a", a],
          ["b", b],
        ]),
        recovery: noRecovery,
        now: clock.now,
      },
    );
    expect(result.outcome).toBe("failure");
    expect(b.calls).toHaveLength(0);
    const critic = result.events.find(
      (e): e is Extract<RunEvent, { type: "critic_note" }> =>
        e.type === "critic_note",
    );
    expect(critic?.text).toContain("timeout");
  });

  it("falls through to next chain entry on max_attempts_per_node = 2", async () => {
    const state = initialRunState("r-attempt");
    const primary = new StubBackend("primary", [
      { kind: "error", reason: "flaky" },
      { kind: "error", reason: "flaky" },
    ]);
    const secondary = new StubBackend("secondary", [
      {
        kind: "success",
        chunks: [chunk(0)],
        final_state: withSummary(state, "s"),
      },
    ]);
    const plan = makePlan(
      [
        {
          id: "n0",
          instruction: "x",
          chosen_backend: "primary",
          fallback_chain: ["secondary"],
          depends_on: [],
        },
      ],
      { max_attempts_per_node: 2 },
    );
    const result = await dispatchPlan(
      { run_id: "r-attempt", plan },
      {
        backends: new Map<string, Backend>([
          ["primary", primary],
          ["secondary", secondary],
        ]),
        recovery: noRecovery,
        now: () => 0,
      },
    );
    expect(result.outcome).toBe("success");
    expect(primary.calls).toHaveLength(2);
    expect(secondary.calls).toHaveLength(1);
  });
});
