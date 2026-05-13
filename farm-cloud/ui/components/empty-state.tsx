import Link from "next/link";

export interface EmptyStateProps {
  title: string;
  body: string;
  ctaHref: string;
  ctaLabel: string;
}

export function EmptyState({ title, body, ctaHref, ctaLabel }: EmptyStateProps) {
  return (
    <section aria-labelledby="empty-state-title" className="empty">
      <h2 id="empty-state-title">{title}</h2>
      <p>{body}</p>
      <Link className="cta" href={ctaHref}>
        {ctaLabel}
      </Link>
    </section>
  );
}
