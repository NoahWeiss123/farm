"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArmViewer } from "@/components/arm-viewer";
import { submitRun, fetchRuns, type RunSummary } from "@/lib/api";

const EXAMPLES = [
  "pick the red block and place it on the cup",
  "pick the blue block and place it on the cup",
  "pick the green block and place it on the cup",
];

export default function LandingPage() {
  const [task, setTask] = useState(EXAMPLES[0]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RunSummary[]>([]);
  const router = useRouter();

  useEffect(() => {
    let live = true;
    const tick = async () => {
      const r = await fetchRuns();
      if (live && r) setRecent(r.slice(0, 5));
    };
    void tick();
    const id = setInterval(tick, 3000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, []);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    const status = await submitRun(task.trim());
    setSubmitting(false);
    if (!status) {
      setError(
        "Couldn't reach the edge daemon. Is `farm serve` running on :8787?",
      );
      return;
    }
    router.push(`/runs/${status.run_id}`);
  };

  return (
    <section className="home">
      <header className="home-hero">
        <h1>FARM</h1>
        <p className="lede">
          Type a task. Watch a 6-DoF UFactory 850 do it in sim. Replay the
          run with every action chunk, safety event, and observation.
        </p>
      </header>

      <div className="home-grid">
        <div className="viewer-card">
          <ArmViewer height={520} />
          <div className="viewer-caption">
            <span className="pulse" /> live world stream — 5 Hz heartbeat
          </div>
        </div>

        <form className="prompt-form" onSubmit={onSubmit}>
          <label htmlFor="task">Ask the arm</label>
          <textarea
            id="task"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            rows={3}
            placeholder="pick the red block and place it on the cup"
          />
          <button type="submit" disabled={submitting || !task.trim()}>
            {submitting ? "submitting..." : "run task"}
          </button>
          {error && <p className="error">{error}</p>}
          <div className="examples">
            <span>examples</span>
            {EXAMPLES.map((ex) => (
              <button
                type="button"
                key={ex}
                className="chip"
                onClick={() => setTask(ex)}
              >
                {ex}
              </button>
            ))}
          </div>

          <div className="recent">
            <h3>recent runs</h3>
            {recent.length === 0 ? (
              <p className="dim">none yet — kick one off above.</p>
            ) : (
              <ul>
                {recent.map((r) => (
                  <li key={r.id}>
                    <Link href={`/runs/${r.id}`}>
                      <span className={`badge state-${r.status}`}>
                        {r.status}
                      </span>
                      <span className="task">{r.task || r.id}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}
