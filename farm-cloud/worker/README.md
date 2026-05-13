# farm-worker

Cloudflare Worker for FARM's cloud side. Hosts the planner, dispatcher
Durable Object, and skill library API.

## Routes

- `GET /` — banner string.
- `GET /healthz` — `{ ok, version, protocol_version }`.
- `POST /v1/plans` — hierarchical task decomposition.
- `GET /v1/runs/:id` — run state from Dispatcher DO.
- `POST /v1/runs/:id/dispatch` — dispatch a plan to a run.
- WebSocket on `/v1/ws` — Edge Agent live channel (obs → action chunks).

## Local

```
bun install
bun run test       # vitest
bun run lint       # tsc --noEmit
bun run dev        # wrangler dev
bun run deploy     # wrangler deploy
```

## Structure

```
src/
  index.ts              # routes + DO export
  env.ts                # bindings
  dispatcher.ts         # Durable Object: per-run state, WS bridge
  planner.ts            # GPT-5 task decomposer
  router/               # capability-card matching + fallback chains
  backends/             # backend adapters (classical, future: pi05, gemini)
  plan_dag.ts           # DAG executor
  run_state.ts          # in-memory run state shape
  fallback.ts           # fallback-on-failure logic
```
