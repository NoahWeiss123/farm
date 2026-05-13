// Cloud-side mirror of farm_shared.RunState (DESIGN.md → State handoff at fallback
// boundaries). Carries the snapshot the next backend needs to pick up from where
// the previous one left off.

export type GripperState = "open" | "closed" | "grasping";

export interface TcpPose {
  x: number;
  y: number;
  z: number;
  roll: number;
  pitch: number;
  yaw: number;
}

export interface Observation {
  // Base64-encoded JPEG frames keyed by camera name. The Edge Agent compresses
  // before sending; the Dispatcher does not decode.
  frames: Record<string, string>;
  joint_state: number[];
  tcp_pose: TcpPose;
  gripper_state: GripperState;
}

export interface RunState {
  run_id: string;
  joint_state: number[];
  tcp_pose: TcpPose;
  gripper_state: GripperState;
  current_node_id: string | null;
  last_completed_chunk_index: number;
  observation: Observation | null;
  critic_summary: string;
}

export function initialRunState(run_id: string): RunState {
  return {
    run_id,
    joint_state: [0, 0, 0, 0, 0, 0],
    tcp_pose: { x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0 },
    gripper_state: "open",
    current_node_id: null,
    last_completed_chunk_index: -1,
    observation: null,
    critic_summary: "",
  };
}

export interface ActionChunk {
  chunk_id: number;
  actions: Array<{
    dx: number;
    dy: number;
    dz: number;
    droll: number;
    dpitch: number;
    dyaw: number;
    gripper: "open" | "close" | "hold" | null;
  }>;
  suggested_dwell_ms: number;
}

export function applyChunk(state: RunState, chunk: ActionChunk): RunState {
  const last = chunk.actions[chunk.actions.length - 1];
  if (last === undefined) {
    return { ...state, last_completed_chunk_index: chunk.chunk_id };
  }
  return {
    ...state,
    tcp_pose: {
      x: state.tcp_pose.x + last.dx,
      y: state.tcp_pose.y + last.dy,
      z: state.tcp_pose.z + last.dz,
      roll: state.tcp_pose.roll + last.droll,
      pitch: state.tcp_pose.pitch + last.dpitch,
      yaw: state.tcp_pose.yaw + last.dyaw,
    },
    gripper_state: gripperAfter(state.gripper_state, last.gripper),
    last_completed_chunk_index: chunk.chunk_id,
  };
}

function gripperAfter(
  current: GripperState,
  command: "open" | "close" | "hold" | null,
): GripperState {
  if (command === "open") return "open";
  if (command === "close") return "closed";
  return current;
}
