import Link from "next/link";

export interface RunCardProps {
  id: string;
  status: string;
}

export function RunCard({ id, status }: RunCardProps) {
  return (
    <article>
      <Link href={`/runs/${id}`}>
        <h3>Run {id}</h3>
        <p>{status}</p>
      </Link>
    </article>
  );
}
