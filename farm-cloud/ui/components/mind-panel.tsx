"use client";

import { useEffect, useState } from "react";
import { subscribeSSE } from "@/lib/sse";
import type { InspectSnapshot } from "@/lib/api";
import { CameraTile } from "@/components/camera-tile";

/**
 * <MindPanel/> — the "what is the arm thinking" surface.
 *
 * Subscribes to /v1/inspect/stream. Shows:
 *  - the task line + current backend (gpt+skills or pi0.5)
 *  - the plan as a step list, with the active step highlighted
 *  - the most recent action label + its raw vector
 *  - the most recent critic note
 *  - two live camera tiles (exterior + wrist) — what the arm "sees"
 *  - a topdown tile that doubles as the "point map"
 *  - the π0.5-shaped observation tensor summary
 */
export function MindPanel({ showDepth = false }: { showDepth?: boolean }) {
  const [snap, setSnap] = useState<InspectSnapshot | null>(null);

  useEffect(() => {
    const es = subscribeSSE("/v1/inspect/stream", (raw) => {
      const ev = raw as { type: string } & InspectSnapshot;
      if (ev?.type === "inspect") setSnap(ev);
    });
    return () => es.close();
  }, []);

  const planNodes = snap?.plan?.nodes ?? [];
  const activeIdx = snap?.active_node_index ?? -1;
  const lastAction = snap?.last_action;
  const obs = snap?.observation;

  return (
    <div className="mind">
      <header className="mind-head">
        <div className="row">
          <span className="kbd">task</span>
          <span className="mind-task">
            {snap?.task ?? <em className="dim">no run yet — submit one →</em>}
          </span>
        </div>
        <div className="row">
          <span className="kbd">policy</span>
          <span className={`pill pill-${snap?.policy ?? "idle"}`}>
            {snap?.policy ?? "idle"}
          </span>
          {snap?.run_id && (
            <span className="run-id mono">{snap.run_id.slice(0, 12)}</span>
          )}
        </div>
      </header>

      <section className="mind-block">
        <h3>plan · {planNodes.length} step{planNodes.length === 1 ? "" : "s"}</h3>
        {planNodes.length === 0 ? (
          <p className="dim">waiting for plan…</p>
        ) : (
          <ol className="plan-list">
            {planNodes.map((n, i) => {
              const state =
                i < activeIdx ? "done" : i === activeIdx ? "active" : "pending";
              return (
                <li key={n.id} className={`plan-step state-${state}`}>
                  <span className="step-bullet">
                    {state === "done" ? "✓" : state === "active" ? "▸" : i + 1}
                  </span>
                  <span className="step-body">
                    <span className="step-name">{prettyInstruction(n.instruction)}</span>
                    <span className="step-meta">
                      {n.id} · {n.backend ?? "sim"}
                    </span>
                  </span>
                </li>
              );
            })}
          </ol>
        )}
        {snap?.plan?.reasoning && (
          <p className="reasoning">
            <span className="kbd">why</span> {snap.plan.reasoning}
          </p>
        )}
      </section>

      <section className="mind-block">
        <h3>now</h3>
        <div className="now-card">
          <span className="now-label">
            {lastAction?.label ?? snap?.last_critic ?? "—"}
          </span>
          {lastAction?.action && (
            <code className="now-action mono">
              {lastAction.action_space === "gripper"
                ? `gripper(${gripperLabel(lastAction.action?.[0])})`
                : `${lastAction.action_space}(${(lastAction.action ?? [])
                    .map((v) => v.toFixed(1))
                    .join(", ")})`}
            </code>
          )}
        </div>
      </section>

      <section className="mind-block">
        <h3>what it sees</h3>
        <div className="cam-grid">
          <CameraTile name="exterior" label="exterior" />
          <CameraTile name="wrist" label="wrist" />
          <CameraTile name="topdown" label="topdown · point map" />
        </div>
        {showDepth && (
          <div className="cam-grid cam-grid-depth">
            <CameraTile name="exterior" variant="depth" label="exterior" />
            <CameraTile name="wrist" variant="depth" label="wrist" />
            <CameraTile name="topdown" variant="depth" label="topdown" />
          </div>
        )}
      </section>

      <section className="mind-block">
        <h3>fed into π0.5</h3>
        {obs ? (
          <dl className="obs-list">
            <div>
              <dt>joint_position (7d)</dt>
              <dd className="mono">
                [{obs.joint_position_7.map((v) => v.toFixed(3)).join(", ")}]
              </dd>
            </div>
            <div>
              <dt>gripper_position</dt>
              <dd className="mono">{obs.gripper_position.toFixed(2)}</dd>
            </div>
            <div>
              <dt>tcp (mm)</dt>
              <dd className="mono">
                [{(obs.tcp_pos_mm ?? []).map((v) => v.toFixed(0)).join(", ")}]
              </dd>
            </div>
            <div>
              <dt>exterior_image_1_left</dt>
              <dd className="mono">224×224 uint8 · {obs.image_urls.exterior}</dd>
            </div>
            <div>
              <dt>wrist_image_left</dt>
              <dd className="mono">224×224 uint8 · {obs.image_urls.wrist}</dd>
            </div>
          </dl>
        ) : (
          <p className="dim">observation will appear once a run starts.</p>
        )}
      </section>
    </div>
  );
}

function prettyInstruction(raw: string): string {
  // The skill executor expects JSON-shaped node instructions like
  // {"skill":"pick_and_place","args":{"source":"red_block","target":"cup"}}.
  // The dashboard reader wants the human-friendly form.
  try {
    const obj = JSON.parse(raw);
    if (obj && typeof obj === "object") {
      const skill = obj.skill ?? "";
      const args = obj.args ?? {};
      const argStr = Object.entries(args)
        .map(([k, v]) => `${k}=${String(v)}`)
        .join(" ");
      return argStr ? `${skill}(${argStr})` : skill;
    }
  } catch {
    /* not JSON */
  }
  return raw;
}

function gripperLabel(code: unknown): string {
  if (typeof code === "string") return code;
  if (code === 0 || code === 0.0) return "open";
  if (code === 1 || code === 1.0) return "closed";
  return String(code);
}

export default MindPanel;
