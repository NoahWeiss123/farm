import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { notFound } from "next/navigation";
import { marked } from "marked";

const DOCS_DIR = path.join(process.cwd(), "..", "..", "docs");

export async function generateStaticParams() {
  const files = await readdir(DOCS_DIR);
  return files
    .filter((f) => f.endsWith(".md"))
    .map((f) => ({ slug: f.replace(/\.md$/, "") }));
}

export default async function DocPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  if (!/^[a-z0-9-]+$/.test(slug)) notFound();

  let raw: string;
  try {
    raw = await readFile(path.join(DOCS_DIR, `${slug}.md`), "utf-8");
  } catch {
    notFound();
  }

  const html = await marked.parse(raw, { gfm: true });
  return (
    <article
      className="prose"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
