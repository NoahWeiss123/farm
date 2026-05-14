# DEPLOY.md — moving FARM from local to Cloudflare

The local build runs entirely against the Python edge daemon (`farm
serve`). When you're ready to deploy, you have three pieces to ship:

1. **Worker** — `farm-cloud/worker/` becomes the planner gateway + the
   run-state proxy that fronts the edge daemon.
2. **Pages** — `farm-cloud/ui/` static export deployed to Cloudflare Pages.
3. **D1 + R2** — skill metadata + artifact storage. (Optional for the
   demo — the local SQLite + filesystem stores already work.)

Nothing about the agentic stack changes. The same `RunSupervisor`, same
`GptPlanner`, same `SkillExecutor` keep running on whatever box hosts
`farm serve`. The worker just stops being optional.

---

## What you need

```
CLOUDFLARE_API_TOKEN=...    # "Edit Cloudflare Workers" template
CLOUDFLARE_ACCOUNT_ID=...   # right sidebar of dash.cloudflare.com
OPENAI_API_KEY=...          # same key as the local build
```

A custom domain is optional. If you skip it you'll get
`https://farm-worker.<your-subdomain>.workers.dev` for free.

---

## Step 1 — push secrets

`.dev.vars` is already gitignored. For production:

```bash
cd farm-cloud/worker
wrangler secret put OPENAI_API_KEY     # paste your key
wrangler secret put OPENAI_MODEL       # optional, defaults to gpt-4o
```

---

## Step 2 — provision D1 + R2

Skip this if you're only fronting the planner and the local daemon still
owns runs/skills.

```bash
# D1: skill metadata (one row per skill, layer, training run, eval)
wrangler d1 create farm-skills
# Copy the database_id from the output and paste it under
# [[d1_databases]] in wrangler.toml. Migrations live in worker/migrations/.

wrangler d1 migrations apply farm-skills

# R2: skill artifacts (generated python, LoRA weights, demo videos)
wrangler r2 bucket create farm-artifacts
# Under [[r2_buckets]] in wrangler.toml:
#   binding = "ARTIFACTS"
#   bucket_name = "farm-artifacts"
```

The worker code only needs to read `env.SKILLS_DB` and `env.ARTIFACTS`
once those bindings exist — see `worker/src/skills/` for the (currently
stubbed) handlers.

---

## Step 3 — deploy the worker

```bash
cd farm-cloud/worker
bun install
bun run test            # 51 tests, ~300 ms
bun run deploy          # wrangler deploy
```

You'll get a `https://farm-worker.<your-subdomain>.workers.dev` URL. Test
it:

```bash
curl -X POST https://farm-worker.<subdomain>.workers.dev/v1/plans \
  -H 'content-type: application/json' \
  -d '{"task":"pick the red block","capability_cards":[]}'
```

---

## Step 4 — point the dashboard at the worker

```bash
cd farm-cloud/ui

# Two env modes are supported:
#   - NEXT_PUBLIC_FARM_API points at the EDGE DAEMON for full agent runs
#     (used during the demo)
#   - NEXT_PUBLIC_FARM_PLANNER points at the WORKER for plan-only requests
#     (used when the edge daemon proxies planning through Cloudflare)

NEXT_PUBLIC_FARM_API=https://farm-worker.<subdomain>.workers.dev \
  bun run build

# For Cloudflare Pages:
bunx wrangler pages deploy out --project-name farm
```

If you want the dashboard on a custom domain (e.g. `farm.example.com`),
add the route in the Pages project settings.

---

## Step 5 — point the edge daemon at Cloudflare for planning

When the worker is up, the edge daemon can offload the OpenAI call to
the worker so the OpenAI key never leaves Cloudflare. Edit your `.env`:

```bash
FARM_PLANNER_URL=https://farm-worker.<subdomain>.workers.dev
```

The `GptPlanner` will check `FARM_PLANNER_URL` first; missing means
local OpenAI is used (current behavior). The migration is one config
change with no code touched.

---

## Per-skill cost meter (when AI Gateway is wired)

AI Gateway caches LLM responses and exposes per-skill spend in the
Cloudflare dashboard. To wire it into the planner:

1. Create an AI Gateway in the Cloudflare dash.
2. Add a service binding to the worker (`ai-gateway` is the conventional name — the planner already looks for it).
3. Re-deploy. Planner calls will route through the gateway automatically.

---

## What stays the same

- Every Python module under `farm-edge-agent/` is unchanged. The agentic
  stack does not care whether the planner endpoint is local or in
  Cloudflare.
- The dashboard's React components are unchanged. `lib/api.ts` reads a
  single env var (`NEXT_PUBLIC_FARM_API`).
- Skill definitions don't move. They're Python modules that ship with
  the edge agent. The cloud only stores their metadata + binary
  artifacts (LoRA weights, training data) once Layer-3 is online.

---

## Rolling back

Delete the worker (`wrangler delete farm-worker`) and the Pages project,
unset `NEXT_PUBLIC_FARM_API` / `FARM_PLANNER_URL`, and the local
demo keeps working exactly as before. There is no point of no return.
