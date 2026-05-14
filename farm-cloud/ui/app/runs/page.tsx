"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { EmptyState } from "@/components/empty-state";
import { fetchRuns, type RunSummary } from "@/lib/api";

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);

  useEffect(() => {
    let live = true;
    const tick = async () => {
      const r = await fetchRuns();
      if (live) setRuns(r ?? []);
    };
    void tick();
    const id = setInterval(tick, 3000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, []);

  if (runs === null) {
    return <p className="dim">loading runs…</p>;
  }

  if (runs.length === 0) {
    return (
      <EmptyState
        title="Run your first task"
        body="Go back to the home page, type a task in the prompt box, and hit run."
        ctaHref="/"
        ctaLabel="Open the dashboard"
      />
    );
  }

  return (
    <section className="runs-list">
      <header>
        <h1>runs</h1>
        <p className="dim">{runs.length} run{runs.length === 1 ? "" : "s"} · auto-refreshing</p>
      </header>
      <ul>
        {runs.map((r) => (
          <li key={r.id}>
            <Link href={`/runs/${r.id}`} className="run-row">
              <span className={`badge state-${r.status}`}>{r.status}</span>
              <span className="task">{r.task || "(no task)"}</span>
              <span className="run-id">{r.id}</span>
              <span className="time dim">
                {new Date(r.submitted_at * 1000).toLocaleTimeString()}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
