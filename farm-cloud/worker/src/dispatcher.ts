// Dispatcher Durable Object. Owns one run from start to finish: walks the
// plan DAG, calls backends, applies the fallback chain on per-node failure,
// and stops the run when the wall-clock budget is exhausted. WebSocket to the
// Edge Agent is held here so a single DO survives the lifetime of the run.

import type { Env } from "./env";
import type { PlanDAG, PlanNode } from "./plan_dag";
import { validatePlan, walkOrder } from "./plan_dag";
import { initialRunState, type RunState } from "./run_state";
import {
  defaultRecoveryRunner,
  runNode,
  type Backend,
  type FallbackEvent,
  type RecoveryRunner,
} from "./fallback";
import { ClassicalBackend } from "./backends/classical";

export type RunEvent =
  | { type: "run_started"; run_id: string; plan_id: string; ts: number }
  | FallbackEvent
  | { type: "critic_note"; node_id: string | null; text: string; ts: number }
  | {
      type: "run_completed";
      run_id: string;
      outcome: "success" | "failure";
      wall_clock_ms: number;
      ts: number;
    };

export interface DispatchDeps {
  backends: Map<string, Backend>;
  recovery: RecoveryRunner;
  now: () => number;
}

export interface DispatchInput {
  run_id: string;
  plan: PlanDAG;
}

export interface DispatchResult {
  run_id: string;
  outcome: "success" | "failure";
  events: RunEvent[];
  final_state: RunState;
}

export async function dispatchPlan(
  input: DispatchInput,
  deps: DispatchDeps,
): Promise<DispatchResult> {
  validatePlan(input.plan);

  const start = deps.now();
  const events: RunEvent[] = [];
  events.push({
    type: "run_started",
    run_id: input.run_id,
    plan_id: input.plan.plan_id,
    ts: start,
  });

  const order = walkOrder(input.plan);
  let state = initialRunState(input.run_id);
  let outcome: "success" | "failure" = "success";

  for (const node of order) {
    const shouldHardStop = (): boolean =>
      deps.now() - start >= input.plan.max_wall_clock_ms;

    if (shouldHardStop()) {
      outcome = "failure";
      events.push({
        type: "critic_note",
        node_id: node.id,
        text: "timeout: max_wall_clock_per_run exceeded before node start",
        ts: deps.now(),
      });
      break;
    }

    const result = await runNode({
      node,
      state,
      backends: deps.backends,
      max_attempts_per_node: input.plan.max_attempts_per_node,
      recovery: deps.recovery,
      shouldHardStop,
    });
    events.push(...result.events);
    state = result.final_state;

    if (result.outcome === "exhausted") {
      outcome = "failure";
      const exhaustedByTimeout = shouldHardStop();
      events.push({
        type: "critic_note",
        node_id: node.id,
        text: exhaustedByTimeout
          ? "timeout: max_wall_clock_per_run exceeded mid-node"
          : `fallback chain exhausted for node ${node.id}: ${result.last_error ?? "unknown error"}`,
        ts: deps.now(),
      });
      break;
    }
  }

  events.push({
    type: "run_completed",
    run_id: input.run_id,
    outcome,
    wall_clock_ms: deps.now() - start,
    ts: deps.now(),
  });

  return { run_id: input.run_id, outcome, events, final_state: state };
}

// In-flight run record held in the DO. Phase-MVP keeps everything in memory;
// Phase-Product writes the same shape to R2 via the Session DO.
interface RunRecord {
  run_id: string;
  outcome: "success" | "failure" | "in_progress";
  events: RunEvent[];
}

export class Dispatcher implements DurableObject {
  private readonly state: DurableObjectState;
  private readonly env: Env;
  private readonly runs = new Map<string, RunRecord>();
  private edgeSocket: WebSocket | null = null;
  private readonly backends: Map<string, Backend>;
  private readonly recovery: RecoveryRunner;

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
    this.backends = defaultBackends();
    this.recovery = defaultRecoveryRunner;
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (
      request.headers.get("Upgrade") === "websocket" &&
      url.pathname === "/ws"
    ) {
      return this.handleWebSocketUpgrade();
    }
    if (request.method === "POST" && url.pathname === "/run") {
      const body = (await request.json()) as DispatchInput;
      const result = await this.start(body);
      return Response.json(result);
    }
    if (request.method === "GET" && url.pathname.startsWith("/runs/")) {
      const id = url.pathname.slice("/runs/".length);
      const record = this.runs.get(id);
      if (record === undefined) {
        return Response.json({ error: "not_found", run_id: id }, { status: 404 });
      }
      return Response.json(record);
    }
    return new Response("not found", { status: 404 });
  }

  async start(input: DispatchInput): Promise<DispatchResult> {
    this.runs.set(input.run_id, {
      run_id: input.run_id,
      outcome: "in_progress",
      events: [],
    });
    const deps: DispatchDeps = {
      backends: this.backends,
      recovery: this.recovery,
      now: () => Date.now(),
    };
    const result = await dispatchPlan(input, deps);
    this.runs.set(input.run_id, {
      run_id: input.run_id,
      outcome: result.outcome,
      events: result.events,
    });
    for (const event of result.events) {
      console.log(JSON.stringify({ run_id: input.run_id, ...event }));
    }
    return result;
  }

  getRun(run_id: string): RunRecord | undefined {
    return this.runs.get(run_id);
  }

  private handleWebSocketUpgrade(): Response {
    const pair = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];
    server.accept();
    this.edgeSocket = server;
    return new Response(null, { status: 101, webSocket: client });
  }
}

export function defaultBackends(): Map<string, Backend> {
  const backends = new Map<string, Backend>();
  const classical = new ClassicalBackend();
  backends.set(classical.id, classical);
  return backends;
}
