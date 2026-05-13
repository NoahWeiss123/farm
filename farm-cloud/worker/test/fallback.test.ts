import { describe, expect, it } from "vitest";
import {
  runNode,
  type Backend,
  type BackendInput,
  type BackendOutcome,
  type RecoveryRunner,
} from "../src/fallback";
import type { PlanNode } from "../src/plan_dag";
import { initialRunState, type ActionChunk, type RunState } from "../src/run_state";

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
      { dx: 0.1, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0, gripper: null },
    ],
  };
}

function nodeWith(chosen: string, fallbacks: string[]): PlanNode {
  return {
    id: "n0",
    instruction: "pick the cube",
    chosen_backend: chosen,
    fallback_chain: fallbacks,
    depends_on: [],
  };
}

const tagState = (s: RunState, tag: string): RunState => ({
  ...s,
  critic_summary: tag,
});

const noRecovery: RecoveryRunner = async (_, s) => s;
const never = (): boolean => false;

describe("runNode happy path", () => {
  it("returns success and emits node_started + action_chunk + node_completed", async () => {
    const state = initialRunState("r1");
    const primary = new StubBackend("primary", [
      {
        kind: "success",
        chunks: [chunk(0)],
        final_state: tagState(state, "primary-done"),
      },
    ]);
    const result = await runNode({
      node: nodeWith("primary", []),
      state,
      backends: new Map([["primary", primary]]),
      max_attempts_per_node: 2,
      recovery: noRecovery,
      shouldHardStop: never,
    });
    expect(result.outcome).toBe("success");
    expect(result.backend_used).toBe("primary");
    expect(result.events.map((e) => e.type)).toEqual([
      "node_started",
      "action_chunk",
      "node_completed",
    ]);
    expect(primary.calls).toHaveLength(1);
    expect(primary.calls[0]?.attempt).toBe(1);
  });
});

describe("fallback to next backend on error", () => {
  it("invokes recovery primitive and hands state to second backend", async () => {
    const state = initialRunState("r2");
    const primaryFailState = tagState(state, "primary-touched");
    const primary = new StubBackend("primary", [
      { kind: "error", reason: "controller_oom" },
      { kind: "error", reason: "controller_oom" },
    ]);
    const secondaryFinal = tagState(primaryFailState, "secondary-done");
    const secondary = new StubBackend("secondary", [
      {
        kind: "success",
        chunks: [chunk(0)],
        final_state: secondaryFinal,
      },
    ]);
    const recoveryCalls: string[] = [];
    const recovery: RecoveryRunner = async (primitive, s) => {
      recoveryCalls.push(primitive);
      return tagState(s, `recovered:${primitive}`);
    };

    const node: PlanNode = {
      id: "n0",
      instruction: "pick",
      chosen_backend: "primary",
      fallback_chain: ["secondary"],
      depends_on: [],
      recovery_primitive: "home",
    };
    const result = await runNode({
      node,
      state,
      backends: new Map<string, Backend>([
        ["primary", primary],
        ["secondary", secondary],
      ]),
      max_attempts_per_node: 2,
      recovery,
      shouldHardStop: never,
    });

    expect(result.outcome).toBe("success");
    expect(result.backend_used).toBe("secondary");
    expect(primary.calls).toHaveLength(2);
    expect(secondary.calls).toHaveLength(1);
    expect(recoveryCalls).toEqual(["home"]);

    const fb = result.events.find((e) => e.type === "fallback_invoked");
    expect(fb).toMatchObject({
      from_backend: "primary",
      to_backend: "secondary",
      trigger: "controller_oom",
    });
    const recoveryEvent = result.events.find(
      (e) => e.type === "recovery_invoked",
    );
    expect(recoveryEvent).toMatchObject({ primitive: "home" });

    expect(secondary.calls[0]?.state.critic_summary).toBe("recovered:home");
    expect(result.final_state.critic_summary).toBe("secondary-done");
  });
});

describe("bounded retry by max_attempts_per_node", () => {
  it("retries exactly max_attempts_per_node times then falls through", async () => {
    const state = initialRunState("r3");
    const primary = new StubBackend("primary", [
      { kind: "error", reason: "transient" },
      { kind: "error", reason: "transient" },
      { kind: "error", reason: "transient" },
    ]);
    const secondary = new StubBackend("secondary", [
      {
        kind: "success",
        chunks: [chunk(0)],
        final_state: tagState(state, "ok"),
      },
    ]);
    const result = await runNode({
      node: nodeWith("primary", ["secondary"]),
      state,
      backends: new Map<string, Backend>([
        ["primary", primary],
        ["secondary", secondary],
      ]),
      max_attempts_per_node: 2,
      recovery: noRecovery,
      shouldHardStop: never,
    });

    expect(result.outcome).toBe("success");
    expect(primary.calls).toHaveLength(2);
    expect(secondary.calls).toHaveLength(1);
  });

  it("returns exhausted when the whole chain fails", async () => {
    const state = initialRunState("r4");
    const primary = new StubBackend("primary", [
      { kind: "error", reason: "boom" },
      { kind: "error", reason: "boom" },
    ]);
    const result = await runNode({
      node: nodeWith("primary", []),
      state,
      backends: new Map([["primary", primary]]),
      max_attempts_per_node: 2,
      recovery: noRecovery,
      shouldHardStop: never,
    });
    expect(result.outcome).toBe("exhausted");
    expect(result.last_error).toBe("boom");
    expect(primary.calls).toHaveLength(2);
  });
});
