# farm ui

Next.js 15 dashboard for FARM. App router, Tailwind 4, strict TypeScript.

## Run

```
bun install
bun run dev      # http://localhost:3000
bun run build
bun run test
```

## Routes

- `/` — landing page with quickstart CTA.
- `/runs` — run history list.
- `/runs/[id]` — run detail with live event stream.
- `/docs` — index of repo `docs/*.md` files.
- `/docs/[slug]` — renders `docs/<slug>.md` as HTML via `marked`.

## Layout

```
app/                # routes
components/         # Nav, EmptyState, RunCard
lib/api.ts          # fetch wrapper to the worker
test/               # vitest + jsdom
```
