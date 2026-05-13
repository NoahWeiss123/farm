# farm-cloud

The hosted half of FARM.

- `worker/` — Cloudflare Worker. Hosts the planner (GPT-5 task decomposer + router), the dispatcher Durable Object (per-run state + WebSocket bridge to the Edge Agent), and the skill library API (read/write against D1 + R2).
- `ui/` — Next.js dashboard. Live run streaming, skill library browser, run record viewer, cost meter, ops dashboard.

Both deploy to a single Cloudflare account. `wrangler dev` works locally.
