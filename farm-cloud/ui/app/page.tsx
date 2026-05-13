import Link from "next/link";

export default function LandingPage() {
  return (
    <section>
      <h1>FARM</h1>
      <p>A hosted harness for robotics foundation models.</p>
      <Link href="/docs/getting-started">Try the quickstart</Link>
    </section>
  );
}
