"use client";

import { use, useEffect, useState } from "react";

export function WarmingUp({ runId }: { runId: string }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const handle = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(handle);
  }, []);

  return (
    <section role="status" aria-live="polite">
      <h2>Warming up...</h2>
      <p>Run {runId}</p>
      <p>
        GPU container is cold-starting. Typical wait is 8–25 seconds.
      </p>
      <p data-testid="elapsed">Elapsed: {elapsed}s</p>
    </section>
  );
}

export default function RunDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return <WarmingUp runId={id} />;
}
