import { EmptyState } from "@/components/empty-state";
import { fetchRuns } from "@/lib/api";
import { RunCard } from "@/components/run-card";

export default async function RunsPage() {
  const runs = await fetchRuns();

  if (!runs || runs.length === 0) {
    return (
      <EmptyState
        title="Run your first task"
        body="Point the Edge Agent at a task and you will see it here."
        ctaHref="/docs/getting-started"
        ctaLabel="Read the getting-started guide"
      />
    );
  }

  return (
    <ul>
      {runs.map((r) => (
        <li key={r.id}>
          <RunCard id={r.id} status={r.status} />
        </li>
      ))}
    </ul>
  );
}
