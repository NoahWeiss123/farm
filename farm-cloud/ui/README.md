# farm ui

Next.js 15 dashboard for FARM. App router, Tailwind 4, strict TypeScript.

## Run

```
bun install
bun run dev      # http://localhost:3000
bun run build
bun run test
```

## Layout

- `app/` — routes. Landing at `/`, run history at `/runs`, run detail at `/runs/[id]`.
- `components/` — `Nav`, `EmptyState`, `RunCard`.
- `lib/api.ts` — fetch wrapper. Returns `null` while the cloud surface is being built.
- `test/` — vitest + jsdom.

The `/runs/[id]` view is the cold-start "warming up..." state from DESIGN.md (Observability). No real data yet.
