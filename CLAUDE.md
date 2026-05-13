# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

FARM (Foundation for Action-Reasoning Models) — a hosted agent harness for robotics foundation models, driving a UFactory 850 6-DOF arm. CS153 final project. The substrate (CLI, drivers, safety, recovery, run records, dispatcher, planner, UI) is built. The current direction (per `IMPLEMENTATION_PLAN.md`) is layering a **self-improving agentic system** on top: hierarchical task decomposition, affordance reasoning, and three-layer skill formation (plan cache → generated code → fine-tuned LoRA).

Source of truth for architecture: `DESIGN.md`. Source of truth for what's next: `IMPLEMENTATION_PLAN.md`.

## Common commands

Python work happens in a venv at `.venv/` at the repo root.

```bash
# one-time setup
source .venv/bin/activate
pip install -e ./farm-shared -e ./farm-edge-agent

# python tests (244 in edge-agent + 21 in shared)
pytest farm-shared/tests farm-edge-agent/tests
pytest farm-edge-agent/tests/safety/test_envelope.py::test_pose_outside_box   # single test
pytest -k "envelope"                                                          # by name pattern

# lint
ruff check .
ruff check --fix .

# worker (Cloudflare, in farm-cloud/worker/)
bun --cwd farm-cloud/worker install
bun --cwd farm-cloud/worker run test    # vitest, 48 tests
bun --cwd farm-cloud/worker run lint    # tsc --noEmit
bun --cwd farm-cloud/worker run dev     # wrangler dev on :8787
bun --cwd farm-cloud/worker run deploy

# UI (Next.js 15 + Tailwind 4, in farm-cloud/ui/)
bun --cwd farm-cloud/ui install
bun --cwd farm-cloud/ui run dev         # http://localhost:3000
bun --cwd farm-cloud/ui run build
bun --cwd farm-cloud/ui run test
```

`farm-edge-agent`'s `pyproject.toml` sets `pythonpath = ["src", "../farm-shared/src"]`, so tests resolve `farm_shared` even without it being installed — but you still need `pip install -e ./farm-shared` for the CLI entry point and IDE resolution.

## Architecture

Three packages plus a cloud half. The split is **edge vs cloud**, joined by a versioned WebSocket protocol.

### Packages

- **`farm-shared/`** — cross-package contracts (Python). The capability-card schema, run-record schema, error catalog (`ErrorCode.E1xxx` enum with format-string templates), protocol versions. Imported by both the Edge Agent and the worker's TypeScript via parallel definitions. This package is normative; if you change a schema here, downstream code must match exactly.

- **`farm-edge-agent/`** — the local half (Python, pip-installable). Owns the control loop, the xArm driver, the LeRobot mock driver, the safety enforcer (envelope/velocity/watchdog/e-stop/singularity), the recovery primitives (home/open_gripper/relocalize/retry_grasp/abort_safely), the JSONL run-record writer, the capability-card validator, the `farm doctor` probes, the CLI (`click`-based, scripted entry point: `farm = "farm_edge_agent.cli.main:main"`), and the public Python API at `farm_edge_agent.client` (re-exported as the `farm` namespace).

- **`farm-cloud/worker/`** — Cloudflare Worker (TypeScript, Hono). Hosts the planner (currently calls Claude Sonnet 4.6 via Anthropic; being switched to OpenAI per A0 in `IMPLEMENTATION_PLAN.md`), the dispatcher Durable Object, the router with confidence/cost scoring (`router/fallback_chain.ts`), and HTTP/WebSocket endpoints.

- **`farm-cloud/ui/`** — Next.js 15 dashboard (App Router, Tailwind 4). Routes: `/`, `/runs`, `/runs/[id]`, `/docs`, `/docs/[slug]` (renders `docs/<slug>.md` via `marked`). Server components read directly from `process.cwd() + "/../../docs"`.

### How a run flows (target shape, partial today)

1. User issues `farm run "<prompt>"`. CLI invokes the **run loop** in the Edge Agent (not yet wired — see A1 in the plan).
2. Edge Agent opens a WebSocket to the dispatcher DO. The protocol handshake (`farm_shared.protocol`) negotiates `protocol_version` and surfaces `FARM-E1006` on mismatch.
3. Dispatcher DO calls the worker's planner; planner returns a plan DAG (`plan_dag.ts`). DAG nodes reference capability cards.
4. Per node, the router picks a backend by `confidence × cost` from the matching capability cards (`router/fallback_chain.ts`).
5. Backend emits `ActionChunk` events back to the Edge Agent over the WebSocket.
6. Edge Agent's safety enforcer validates each chunk against the envelope/velocity/singularity rules; rejected chunks invoke recovery primitives.
7. Approved chunks go to the driver. Run record writer appends every event to `~/.farm/runs/<run_id>/record.jsonl`.

Several pieces are real but **orphaned** — written, tested, no callers: `SafetyEnforcer`, `RunRecordWriter`, recovery primitives. They're waiting for the run loop (A1) to be built.

## Key invariants

- **The capability-card schema** in `farm-shared/schemas/capability_card.v1.json` is the contract between backends and the router. Changing it requires updating both the validator (`farm_edge_agent.capability_cards.validator`) and consumers.
- **`ErrorCode` in `farm_shared.errors`** uses an `Enum` whose values are `_Spec` dataclasses (code, template, docs_url_slug). Use `format_error(code, **slots)` to render, never f-string the enum directly. The enum has both `Exxxx` names (canonical) and module-level symbolic aliases (`NO_CAMERA`, etc.) — call sites must use `ErrorCode.Exxxx`, not `ErrorCode.NO_CAMERA`.
- **Two `FarmError` classes coexist on purpose**: `farm_edge_agent.errors.FarmError` (structured, with severity/exit_code/docs_url, set slots as attributes) and `farm_edge_agent.client.FarmError` (simple Exception for the public Python API). Don't merge them.
- **Run records are append-only JSONL**, one file per run, with HMAC signing planned (G3). Replay must be deterministic from the record alone — don't write timestamps as the source of truth.
- **Tests are deterministic**. No `sleep()`, no unseeded randomness. Network/hardware tests must be in `*_integration_test.py` files.

## Commit conventions (enforced by hook)

The repo has `core.hooksPath = .githooks` and a `prepare-commit-msg` hook that **automatically** strips AI tells and normalizes commit messages. Don't fight it; write messages that don't need normalizing.

- Lowercase title, present-tense imperative, under 50 chars.
- **No prefixes.** No `feat:`, `fix:`, `chore:`, `docs:`, `refactor:` — the hook strips them.
- No `Co-Authored-By: Claude`, no `Generated with [Claude Code]`, no robot emojis — the hook strips them.
- Body optional, 1–2 lines max, no marketing language ("comprehensive", "robust"), no hedging ("this should", "this attempts").
- Good: `add cli scaffold` / `wire xarm driver` / `fix gripper state on home`
- Bad: `feat(cli): add comprehensive CLI scaffolding`

The only allowed prefix is `WIP:` for incomplete work the orchestrator should retry.

## Local-only orchestration layer (gitignored)

These files live on disk but never ship to GitHub:

- `bin/orchestrate`, `bin/run-task`, `bin/ledger` — bash orchestration scripts (bash 3.2 compatible for macOS).
- `tasks/_ledger.json`, `tasks/NNN-<slug>.md` — task spec files for the orchestrator.
- `AGENTS.md` — rules agents follow when run by the orchestrator.
- `LAUNCH.md` — launch instructions for the overnight build.
- `.githooks/prepare-commit-msg` — the commit-message hook.
- `worktrees/`, `logs/` — per-run scratch space.

When running the orchestrator, it marks tasks `in_progress` **before** spawning the `run-task` subprocess to prevent a race where the next poll loop sees the same task as still pending.

## Known stubs to avoid recreating

These CLI commands print `[FARM] not implemented yet` on purpose. They depend on the run loop being built (A1). Don't accidentally "fix" them with placeholder logic:

- `farm run`, `farm start`, `farm calibrate`, `farm verify`, `farm quickstart`, `farm login`

`farm doctor real-arm` is similarly stubbed pending C8.

## Scope discipline

If your task touches Python, keep changes in `farm-edge-agent/` or `farm-shared/`. If TypeScript, `farm-cloud/`. The cross-package contracts in `farm-shared/` are the only thing both sides import — if you add a new field there, you must update both the Python schema and the TS consumer in the same change.

Most tasks should *not* touch the dispatcher DO, the planner, the safety enforcer, or the capability-card schema. If you find yourself there, double-check that's actually in scope.
