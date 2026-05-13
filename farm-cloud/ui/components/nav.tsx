import Link from "next/link";

export function Nav() {
  return (
    <nav aria-label="primary">
      <Link href="/">FARM</Link>
      <Link href="/runs">Runs</Link>
      <Link href="/docs">Docs</Link>
    </nav>
  );
}
