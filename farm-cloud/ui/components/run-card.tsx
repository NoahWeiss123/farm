import Link from "next/link";

export interface RunCardProps {
  id: string;
  status: string;
}

export function RunCard({ id, status }: RunCardProps) {
  return (
    <Link href={`/runs/${id}`} className="card">
      <div className="card-title">Run {id}</div>
      <div className="card-meta">{status}</div>
    </Link>
  );
}
