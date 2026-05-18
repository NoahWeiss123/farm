"use client";

import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArmViewer } from "@/components/arm-viewer";
import { MindPanel } from "@/components/mind-panel";
import { fetchRun, type RunDetail, type RunEvent } from "@/lib/api";
import { subscribeSSE } from "@/lib/sse";

interface EventListProps {
  events: RunEvent[];
}

function EventList({ events }: EventListProps) {
  return (
    <ol className="event-list">
      {events.map((e, i) => {
        const t = new Date(e.ts * 1000).toLocaleTimeString();
        let summary = "";
        switch (e.type) {
          case "run_started":
            summary = `task: "${(e.data.task ?? "") as string}"`;
            break;
          case "plan_emitted":
            summary = `plan ${e.data.plan_id} (${(e.data.nodes as unknown[])?.length ?? 0} nodes)`;
            break;
          case "node_started":
            summary = `node ${e.data.node_id} on ${e.data.backend}`;
            break;
          case "action_chunk": {
            const action = (e.data.action ?? []) as number[];
            const space = (e.data.action_space ?? "") as string;
            const label = (e.data.label as string) || "";
            const head = label ? `${label} · ` : "";
            summary = `${head}${space} [${action.map((a) => a.toFixed(1)).join(", ")}]`;
            break;
          }
          case "obs_chunk":
            summary = `obs (${((e.data.joint_state ?? []) as number[]).length} joints)`;
            break;
          case "safety_event":
            summary = `${e.data.kind}: ${e.data.detail}`;
            break;
          case "critic_note":
            summary = (e.data.text ?? "") as string;
            break;
          case "recovery_invoked":
            summary = `→ ${e.data.primitive}`;
            break;
          case "node_completed":
            summary = `→ ${e.data.outcome}`;
            break;
          case "run_completed":
            summary = `${e.data.outcome} in ${(e.data.wall_clock_s as number)?.toFixed(2)}s`;
            break;
          default:
            summary = JSON.stringify(e.data);
        }
        return (
          <li key={i} className={`ev ev-${e.type}`}>
            <span className="ts">{t}</span>
            <span className="kind">{e.type}</span>
            <span className="summary">{summary}</span>
          </li>
        );
      })}
    </ol>
  );
}

export default function RunDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [liveEvents, setLiveEvents] = useState<RunEvent[]>([]);

  useEffect(() => {
    let alive = true;
    fetchRun(id).then((d) => {
      if (alive && d) setDetail(d);
    });
    const es = subscribeSSE(`/v1/runs/${encodeURIComponent(id)}/events`, (raw) => {
      const ev = raw as RunEvent;
      if (!ev || typeof ev.type !== "string") return;
      setLiveEvents((prev) => [...prev, ev]);
    });
    return () => {
      alive = false;
      es.close();
    };
  }, [id]);

  const events = useMemo(() => {
    const seen = new Set<string>();
    const all: RunEvent[] = [];
    for (const e of liveEvents) {
      const key = `${e.ts}-${e.type}-${JSON.stringify(e.data).slice(0, 80)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      all.push(e);
    }
    if (all.length > 0) return all;
    return detail?.events ?? [];
  }, [liveEvents, detail]);

  const status = useMemo(() => {
    const base = detail?.status;
    if (!base) return base;
    let merged = { ...base };
    for (const ev of liveEvents) {
      if (ev.type === "run_started") {
        merged = { ...merged, state: "running" };
      } else if (ev.type === "plan_emitted") {
        merged = { ...merged, plan_id: (ev.data.plan_id as string) ?? null };
      } else if (ev.type === "safety_event") {
        merged = { ...merged, safety_events: (merged.safety_events ?? 0) + 1 };
      } else if (ev.type === "run_completed") {
        const outcome = (ev.data.outcome as string) ?? "succeeded";
        merged = { ...merged, state: outcome, outcome, completed_at: ev.ts };
      }
    }
    return merged;
  }, [detail, liveEvents]);

  return (
    <section className="full-bleed run-detail">
      <header className="run-header">
        <Link href="/runs" className="back">← all runs</Link>
        <div className="header-main">
          <h1>{status?.task || id}</h1>
          <div className="meta">
            <span className={`badge state-${status?.state ?? "queued"}`}>
              {status?.state ?? "queued"}
            </span>
            <span className="run-id">{id}</span>
            {status?.plan_id && <span className="plan-id">{status.plan_id}</span>}
            {status?.safety_events !== undefined && (
              <span>safety: {status.safety_events}</span>
            )}
          </div>
        </div>
      </header>

      <div className="run-viewport run-viewport-3col">
        <div className="viewer-pane">
          <ArmViewer height="100%" />
        </div>
        <aside className="mind-pane">
          <MindPanel />
        </aside>
        <aside className="timeline-pane">
          <h2>timeline · {events.length} events</h2>
          <EventList events={events} />
        </aside>
      </div>
    </section>
  );
}
