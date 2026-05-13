# farm-worker

Cloudflare Worker for FARM's cloud side. Phase-MVP collapses the Planner, Dispatcher,
and Session components into this single Worker. Task 017 splits them out.

## Routes

- `GET /` — banner string.
- `GET /healthz` — `{ ok, version, protocol_version }`.
- `POST /v1/plans` — planner stub (501).
- `POST /v1/runs/:id/dispatch` — dispatcher stub (501).
- `GET /v1/runs/:id` — session stub (501).

## Local

```
bun install
bun run test       # vitest
bun run lint       # tsc --noEmit
bun run dev        # wrangler dev
bun run deploy     # wrangler deploy
```
