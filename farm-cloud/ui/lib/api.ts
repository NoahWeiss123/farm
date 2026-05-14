/**
 * Client for the FARM edge daemon (and, once deployed, the Cloudflare worker
 * that proxies it). Set NEXT_PUBLIC_FARM_API to point at the daemon — defaults
 * to localhost:8787 which matches `farm serve` out of the box.
 */

export const FARM_API =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_FARM_API
    ? process.env.NEXT_PUBLIC_FARM_API.replace(/\/$/, "")
    : "http://127.0.0.1:8787";

export interface RunStatus {
  run_id: string;
  task: string;
  state: string;
  submitted_at: number;
  started_at: number | null;
  completed_at: number | null;
  outcome: string | null;
  error: string | null;
  plan_id: string | null;
  safety_events: number;
}

export interface RunSummary {
  id: string;
  status: string;
  task: string;
  outcome: string | null;
  submitted_at: number;
}

export interface RunDetail {
  status: RunStatus;
  events: RunEvent[];
}

export interface RunEvent {
  ts: number;
  type: string;
  data: Record<string, unknown>;
  run_id?: string;
}

export interface SceneProp {
  id: string;
  shape: string;
  size: number[];
  pos: [number, number, number];
  rgba?: [number, number, number, number];
}

export interface SceneSpec {
  name: string;
  props: SceneProp[];
}

export interface WorldSnapshot {
  joints: number[];
  tcp_pos_m: [number, number, number];
  tcp_quat: [number, number, number, number];
  gripper: string;
  props: Record<string, { pos: [number, number, number]; quat: number[] }>;
  scene?: string;
}

const FETCH_OPTS: RequestInit = { cache: "no-store" };

export async function fetchRuns(): Promise<RunSummary[] | null> {
  try {
    const r = await fetch(`${FARM_API}/v1/runs`, FETCH_OPTS);
    if (!r.ok) return null;
    const json = (await r.json()) as { runs: RunStatus[] };
    return json.runs.map((s) => ({
      id: s.run_id,
      status: s.state,
      task: s.task,
      outcome: s.outcome,
      submitted_at: s.submitted_at,
    }));
  } catch {
    return null;
  }
}

export async function fetchRun(id: string): Promise<RunDetail | null> {
  try {
    const r = await fetch(`${FARM_API}/v1/runs/${encodeURIComponent(id)}`, FETCH_OPTS);
    if (!r.ok) return null;
    return (await r.json()) as RunDetail;
  } catch {
    return null;
  }
}

export async function submitRun(task: string): Promise<RunStatus | null> {
  try {
    const r = await fetch(`${FARM_API}/v1/runs`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task }),
    });
    if (!r.ok) return null;
    return (await r.json()) as RunStatus;
  } catch {
    return null;
  }
}

export async function fetchScene(): Promise<SceneSpec | null> {
  try {
    const r = await fetch(`${FARM_API}/v1/scene`, FETCH_OPTS);
    if (!r.ok) return null;
    return (await r.json()) as SceneSpec;
  } catch {
    return null;
  }
}

export async function fetchWorld(): Promise<WorldSnapshot | null> {
  try {
    const r = await fetch(`${FARM_API}/v1/world`, FETCH_OPTS);
    if (!r.ok) return null;
    return (await r.json()) as WorldSnapshot;
  } catch {
    return null;
  }
}
