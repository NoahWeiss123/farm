// Stub adapter for the local classical planner. Real implementation will call
// into the Python classical backend hosted by the Edge Agent over the WS. For
// Phase-MVP this returns deterministic canned chunks so the rest of the
// pipeline can be exercised end-to-end.

import type { ActionChunk, RunState } from "../run_state";
import { applyChunk } from "../run_state";
import type { Backend, BackendInput, BackendOutcome } from "../fallback";

export const CLASSICAL_BACKEND_ID = "classical-planner";

export class ClassicalBackend implements Backend {
  readonly id = CLASSICAL_BACKEND_ID;

  async run(input: BackendInput): Promise<BackendOutcome> {
    const chunk = cannedChunkFor(input.state.last_completed_chunk_index + 1);
    const final_state = applyChunk(input.state, chunk);
    return {
      kind: "success",
      chunks: [chunk],
      final_state: {
        ...final_state,
        critic_summary: `classical-planner completed node ${input.node.id}`,
      },
    };
  }
}

function cannedChunkFor(chunk_id: number): ActionChunk {
  return {
    chunk_id,
    suggested_dwell_ms: 250,
    actions: [
      { dx: 0.01, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0, gripper: null },
      { dx: 0.01, dy: 0, dz: 0, droll: 0, dpitch: 0, dyaw: 0, gripper: null },
    ],
  };
}

export function classicalCannedFinalState(
  initial: RunState,
  chunk_id: number,
): RunState {
  return applyChunk(initial, cannedChunkFor(chunk_id));
}
