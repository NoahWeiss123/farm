"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArmViewer } from "@/components/arm-viewer";
import { submitRun, fetchRuns, type RunSummary } from "@/lib/api";

const EXAMPLES: string[] = [
  "pick the red block and place it on the cup",
  "pick the blue block and place it on the cup",
  "pick the green block and place it on the cup",
  "stack the red block on the blue block",
];

export default function LandingPage() {
  const [task, setTask] = useState<string>(EXAMPLES[0] ?? "");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RunSummary[]>([]);
  const router = useRouter();

  useEffect(() => {
    let live = true;
    const tick = async () => {
      const r = await fetchRuns();
      if (live && r) setRecent(r.slice(0, 6));
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
    <section className="full-bleed home">
      <div className="home-viewport">
        <ArmViewer height="100%" />
        <div className="viewer-overlay">
          <div className="overlay-top">
            <h1>FARM</h1>
            <p className="tagline">
              UFactory 850 · MuJoCo physics · GPT-decomposed tasks
            </p>
          </div>
          <div className="overlay-bottom">
            <span className="pulse" /> live world stream
          </div>
        </div>
      </div>

      <aside className="home-side">
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
        </form>

        <section className="recent">
          <h3>recent runs</h3>
          {recent.length === 0 ? (
            <p className="dim">none yet — kick one off above.</p>
          ) : (
            <ul>
              {recent.map((r) => (
                <li key={r.id}>
                  <Link href={`/runs/${r.id}`}>
                    <span className={`badge state-${r.status}`}>{r.status}</span>
                    <span className="task">{r.task || r.id}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </aside>
    </section>
  );
}
