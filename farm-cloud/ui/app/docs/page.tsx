import Link from "next/link";
import { readdir } from "node:fs/promises";
import path from "node:path";

const DOCS_DIR = path.join(process.cwd(), "..", "..", "docs");

function titleize(slug: string): string {
  return slug
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export default async function DocsIndex() {
  const files = await readdir(DOCS_DIR);
  const slugs = files
    .filter((f) => f.endsWith(".md"))
    .map((f) => f.replace(/\.md$/, ""))
    .sort();

  return (
    <section>
      <h1>Docs</h1>
      <p>Reference and runbooks for FARM.</p>
      <ul>
        {slugs.map((slug) => (
          <li key={slug}>
            <Link href={`/docs/${slug}`}>{titleize(slug)}</Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
