// Fallback chain executor (DESIGN.md → Fallback chain + State handoff).
// Walks the runtime chain for a single node, retries up to
// max_attempts_per_node per backend, and invokes a recovery primitive between
// backends so the new one starts from a known pose.

import type { PlanNode, RecoveryPrimitive } from "./plan_dag";
import { runtimeChainFor } from "./plan_dag";
import type { ActionChunk, RunState } from "./run_state";

export interface BackendInput {
  node: PlanNode;
  state: RunState;
  instruction: string;
  attempt: number;
}

export type BackendOutcome =
  | { kind: "success"; chunks: ActionChunk[]; final_state: RunState }
  | { kind: "error"; reason: string };

export interface Backend {
  readonly id: string;
  run(input: BackendInput): Promise<BackendOutcome>;
}

export type RecoveryRunner = (
  primitive: RecoveryPrimitive,
  state: RunState,
) => Promise<RunState>;

export type FallbackEvent =
  | { type: "node_started"; node_id: string; backend: string }
  | { type: "action_chunk"; node_id: string; chunk: ActionChunk }
  | {
      type: "fallback_invoked";
      node_id: string;
      from_backend: string;
      to_backend: string;
      trigger: string;
    }
  | {
      type: "recovery_invoked";
      node_id: string;
      primitive: RecoveryPrimitive;
    }
  | { type: "node_completed"; node_id: string; outcome: "success" | "exhausted" };

export interface FallbackResult {
  outcome: "success" | "exhausted";
  final_state: RunState;
  backend_used: string | null;
  events: FallbackEvent[];
  last_error: string | null;
}

export interface RunNodeOptions {
  node: PlanNode;
  state: RunState;
  backends: Map<string, Backend>;
  max_attempts_per_node: number;
  recovery: RecoveryRunner;
  // Returns elapsed ms since run start. If it exceeds the run budget the
  // executor stops walking the chain and returns `exhausted` with reason
  // "timeout".
  shouldHardStop: () => boolean;
}

export async function runNode(
  opts: RunNodeOptions,
): Promise<FallbackResult> {
  const chain = runtimeChainFor(opts.node);
  const events: FallbackEvent[] = [];
  let state: RunState = { ...opts.state, current_node_id: opts.node.id };
  let previousBackend: string | null = null;
  let lastError: string | null = null;

  for (const backendId of chain) {
    if (opts.shouldHardStop()) {
      events.push({
        type: "node_completed",
        node_id: opts.node.id,
        outcome: "exhausted",
      });
      return {
        outcome: "exhausted",
        final_state: state,
        backend_used: previousBackend,
        events,
        last_error: lastError ?? "timeout",
      };
    }

    const backend = opts.backends.get(backendId);
    if (backend === undefined) {
      lastError = `backend ${backendId} not registered`;
      previousBackend = backendId;
      continue;
    }

    if (previousBackend !== null && previousBackend !== backendId) {
      const primitive = opts.node.recovery_primitive ?? "home";
      events.push({
        type: "recovery_invoked",
        node_id: opts.node.id,
        primitive,
      });
      state = await opts.recovery(primitive, state);
      events.push({
        type: "fallback_invoked",
        node_id: opts.node.id,
        from_backend: previousBackend,
        to_backend: backendId,
        trigger: lastError ?? "error",
      });
    } else {
      events.push({
        type: "node_started",
        node_id: opts.node.id,
        backend: backendId,
      });
    }

    let attempts = 0;
    let success = false;
    while (attempts < opts.max_attempts_per_node) {
      if (opts.shouldHardStop()) break;
      attempts++;
      const outcome = await backend.run({
        node: opts.node,
        state,
        instruction: opts.node.instruction,
        attempt: attempts,
      });
      if (outcome.kind === "success") {
        for (const chunk of outcome.chunks) {
          events.push({
            type: "action_chunk",
            node_id: opts.node.id,
            chunk,
          });
        }
        state = outcome.final_state;
        success = true;
        break;
      }
      lastError = outcome.reason;
    }

    if (success) {
      events.push({
        type: "node_completed",
        node_id: opts.node.id,
        outcome: "success",
      });
      return {
        outcome: "success",
        final_state: state,
        backend_used: backendId,
        events,
        last_error: null,
      };
    }

    previousBackend = backendId;
  }

  events.push({
    type: "node_completed",
    node_id: opts.node.id,
    outcome: "exhausted",
  });
  return {
    outcome: "exhausted",
    final_state: state,
    backend_used: previousBackend,
    events,
    last_error: lastError,
  };
}

export const defaultRecoveryRunner: RecoveryRunner = async (_, state) => {
  // Stub: returns the input state untouched. Real implementation issues the
  // primitive over the WS and waits for the Edge Agent to confirm.
  return state;
};
