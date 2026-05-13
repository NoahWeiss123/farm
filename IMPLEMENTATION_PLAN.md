# Implementation plan

A self-improving robotics agent that decomposes tasks, reasons about
environmental affordances, and forms reusable skills across three layers.
Built on the FARM scaffold.

## Locked-in decisions

| Decision | Choice |
|---|---|
| Scope | Bundle δ — agentic skill-forming system, real-robot, hosted, fine-tuned |
| LLM provider | OpenAI (GPT-5 + GPT-5V) via `OPENAI_API_KEY` |
| Perception | GPT-5V for scene understanding |
| Demo tiers | All four (basics, multi-step, env-reasoning, attempted cube) |
| Skill formation | All three layers (plan cache + generated code + fine-tuned LoRA) |
| Skill library | Cloudflare D1 (metadata) + R2 (artifacts) |
| Cube hardware | Agent figures out interaction; no custom gripper |
| Hosting | Deploy to Cloudflare |
| Inference | Modal (for π0.5 LoRA) |
| Work mode | Hybrid (orchestrator for foundation + skill engine, serial for robot work) |
| Budget | No cap |

## What you're building (one paragraph)

A user gives the agent a goal in natural language. GPT-5 decomposes it into
sub-tasks. For each sub-task, the agent first searches the **Skill Library**
for a matching skill. If found, it executes the skill (cheap). If not, the
**Affordance Reasoner** describes what physical interactions are possible
given the arm + gripper + camera + scene, and the **Code-as-Policy** executor
generates Python that calls primitive actions (`grasp`, `move_to`, `push`,
`rotate_wrist`). The code runs under the existing **Safety Enforcer**. On
success, the **Skill Compiler** distills the run into a reusable skill at
three levels: cached plan, parameterized code, and (after N successful
demos) a LoRA-fine-tuned π0.5 policy.

---

## Step 0: Credentials I need from you

Put in `.env` at repo root (gitignored).

```bash
# Required immediately — drives the planner, decomposer, code-gen, vision
OPENAI_API_KEY=sk-...

# Required for cloud deploy + skill library
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_ACCOUNT_ID=...

# Required for fine-tuning
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...

# Optional
HUGGINGFACE_TOKEN=...     # publish dataset to HF
SENTRY_DSN=...            # error reporting
```

Where to get them:

| Key | Where |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com → API Keys |
| `CLOUDFLARE_API_TOKEN` | dash.cloudflare.com → My Profile → API Tokens → "Edit Cloudflare Workers" template |
| `CLOUDFLARE_ACCOUNT_ID` | dash.cloudflare.com → right sidebar |
| `MODAL_TOKEN_*` | `pip install modal && modal token new` |
| `HUGGINGFACE_TOKEN` | huggingface.co/settings/tokens |

## Step 1: Local + physical setup

### Local

- [x] Python 3.12, bun ≥1.0 installed
- [x] Repo cloned, `.venv` ready, all tests passing
- [ ] `brew install ffmpeg`
- [ ] `pip install opencv-python xarm-python-sdk modal openai`
- [ ] `.env` created with `OPENAI_API_KEY` at minimum

### Physical

- [ ] UFactory 850 powered, on LAN, pingable, IP noted
- [ ] At least one USB-3 camera plugged in
- [ ] Second camera (overhead) — strongly recommended
- [ ] E-stop button on a cable, accessible
- [ ] Workspace ~60×60cm cleared, two desk lamps
- [ ] Props: 4+ colored blocks, 1 cup, 1 drawer or box with handle, 1 heavier "blocker" object, 1 Rubik's cube
- [ ] Printed 7×6 checkerboard at 25mm squares on stiff backing

---

## Step 2: Feature menu

Every feature is fully functional (real code, tests, integration-verified).
Ordered roughly by dependency. Tick what you want; defaults are
auto-ticked because they're load-bearing for the vision you described.

### A. Foundation

- [x] **A0. Swap planner LLM to OpenAI** — auto-confirmed.
- [ ] **A1. RunLoop core** — async obs → planner → safety → driver → record.
- [ ] **A2. Edge↔cloud WebSocket pump** — Dispatcher DO reads/writes messages.
- [ ] **A3. WS transport for Python client** — `farm.Client` actually connects.
- [ ] **A4. Safety enforcer wired into RunLoop**.
- [ ] **A5. Recovery primitives wired into failure handler**.
- [ ] **A6. Mock observation source** — synthetic frames for dev without cameras.
- [ ] **A7. Worker `/v1/runs` REST endpoints**.
- [ ] **A8. UI api wiring** — `fetchRuns`/`fetchRun` hit the worker.
- [ ] **A9. Live run streaming in UI**.
- [ ] **A11. Strict CI** — remove all `|| true`.

### M. Perception & world model

- [ ] **M1. GPT-5V scene description** — frames → structured scene graph.
- [ ] **M2. Object tracking across frames** — persistent IDs.
- [ ] **M3. Affordance reasoner** — given scene + tools, return feasible primitive actions per object.
- [ ] **M4. World state representation** — symbolic graph (objects, relations, history) for the decomposer.
- [ ] **M5. Change detection** — diff state before/after action to verify outcome.

### L. Skill formation engine

- [ ] **L1. Skill library schema (D1)** — skills, demos, evals, parent skills.
- [ ] **L2. Skill artifact storage (R2)** — code, weights, demo videos, eval runs.
- [ ] **L3. Hierarchical decomposer** — recursive GPT-5 call that breaks goals into sub-tasks; each sub-task matches an existing skill or is novel.
- [ ] **L4. Skill retriever** — semantic search by description embedding; returns top-k.
- [ ] **L5. Code-as-policy executor** — LLM-generated Python in sandboxed subprocess; AST-checked.
- [ ] **L6. Action primitives library** — `grasp`, `release`, `move_to`, `push`, `rotate_wrist`, `brace`, `lift`, `lower`, `slide`. Each backed by xArm motion.
- [ ] **L7. Skill compiler — Layer 1 (plan cache)** — store exact action sequence keyed by (goal, scene-hash).
- [ ] **L8. Skill compiler — Layer 2 (generated code)** — turn the successful execution into a parameterized Python function the LLM produces, stored as a callable skill.
- [ ] **L9. Skill compiler — Layer 3 (LoRA)** — after N successful demos (default 10), kick off a Modal LoRA fine-tune of π0.5; register as a new backend with confidence boost.
- [ ] **L10. Cost-aware skill router** — choose layer (L3 > L2 > L1 > novel) based on confidence + latency budget.

### C. Real robot integration

- [ ] **C1. xArm SDK verify** — confirm shim matches real SDK; fix BEFORE any motion.
- [ ] **C2. Real OpenCV camera capture**.
- [ ] **C3. Multi-camera time sync**.
- [ ] **C4. Camera intrinsics calibration** — `farm calibrate cameras`.
- [ ] **C5. Hand-eye extrinsics calibration** — `farm calibrate hand-eye`.
- [ ] **C6. Workspace envelope from real geometry**.
- [ ] **C7. Gripper control with force feedback**.
- [ ] **C8. `farm doctor real-arm` interactive walkthrough**.

### N. Demo tasks

**Tier 1 — basics**

- [ ] **N1.1. Pick + place** — red block in cup, end-to-end through the agentic stack.
- [ ] **N1.2. Stack 3 blocks** — multi-pick, ordered.
- [ ] **N1.3. Color sort** — language grounding.

**Tier 2 — multi-step planning**

- [ ] **N2.1. Tower of Hanoi (3 disks)** — hierarchical decomposition; second attempt visibly faster (skill formation working).
- [ ] **N2.2. Build the structure from this image**.
- [ ] **N2.3. Mirror this pattern**.

**Tier 3 — environment reasoning**

- [ ] **N3.1. Open the drawer** — identify handle, grasp, pull.
- [ ] **N3.2. Slide the book under the laptop** — use the gap.
- [ ] **N3.3. Move the blocker first** — sequenced obstacle clearing.
- [ ] **N3.4. Pick this without the gripper closing** — alternate-grasp reasoning (pinch by corner, push-and-lift).

**Tier 4 — attempted Rubik's cube**

- [ ] **N4.1. Read cube state** — GPT-5V identifies all 54 stickers as the cube is rotated in view.
- [ ] **N4.2. Generate solving algorithm** — Kociemba via off-the-shelf library.
- [ ] **N4.3. Attempt face rotations** — system explores creative interactions (brace + wrist rotate, table-edge bracing, etc).
- [ ] **N4.4. Failure analysis + report** — agent identifies hardware limitations, explains why physical execution fails, reports as structured limitation in the run record.

### D. Cloudflare production deploy

- [ ] **D1. R2 bucket bindings** — run records, frames, videos, skill artifacts.
- [ ] **D2. D1 schema + bindings** — runs, skills, evals, demos.
- [ ] **D3. Production worker deploy**.
- [ ] **D4. AI Gateway for OpenAI** — aggressive caching, ~50% cost cut.
- [ ] **D5. UI deployed to Cloudflare Pages**.
- [ ] **D6. Custom domain (optional)**.
- [ ] **D7. Analytics Engine metrics** — per-skill success rate, layer-hit rate.
- [ ] **D8. Workers Logpush**.

### F. Fine-tuning pipeline

- [ ] **F1. Demo capture mode** — every successful run saves training-ready data.
- [ ] **F2. LeRobot dataset export from skill demos**.
- [ ] **F3. HuggingFace dataset upload** (optional).
- [ ] **F4. openpi LoRA training on Modal** — auto-kicked when a skill hits N demos.
- [ ] **F5. Model upload + skill registration** — trained LoRA becomes L3 backend for that skill.
- [ ] **F6. Eval harness — held-out test runs** — measure skill success rate before/after fine-tune.
- [ ] **F7. A/B test infrastructure** — route X% of runs to candidate model.
- [ ] **F8. Demo curation UI** — review captured demos, flag bad ones.

### E. Additional model adapters

- [ ] **E5. Gemini Robotics 1.5 adapter** — only if you get API access.
- [ ] **E6. BYO model registration endpoint** — REST endpoint to register a custom backend.
- [ ] **E7. Critic loop** — second LLM reviews plans, can request replan.

### B. CLI completeness

- [ ] **B1. `farm run "<prompt>"`** — execute end-to-end against the agentic stack.
- [ ] **B2. `farm start`** — long-running daemon, persistent WS.
- [ ] **B4. `farm login --dev`** — local key-paste flow.
- [ ] **B5. `farm quickstart`** — interactive onboarding (config init → doctor → first run).
- [ ] **B6. `farm verify <run-id>`** — verify signature + replay.
- [ ] **B7. `farm calibrate`** — interactive hand-eye calibration.
- [ ] **B8. `farm eval <suite>`** — run a fixture set, report pass rate.

### G. Run records & observability

- [ ] **G1. Video recording during runs**.
- [ ] **G2. Video playback in UI**, scrubber synced to action chunks.
- [ ] **G3. HMAC signing of run records** — `farm verify` detects tampering.
- [ ] **G4. Run record viewer / replayer** — play back any record visually.
- [ ] **G5. Cost meter per run + per skill**.
- [ ] **G6. Skill performance dashboard** — layer-hit rate, success rate, $ over time.
- [ ] **G7. Sentry error reporting**.
- [ ] **G8. Audit log** — append-only log of who-ran-what-when.

### I. UI polish

- [ ] **I1. Real-time ops dashboard at `/ops`**.
- [ ] **I2. Skill library browser** — view skills, drill into demos.
- [ ] **I3. Model registry UI**.
- [ ] **I5. Settings page**.
- [ ] **I6. Mobile-responsive layouts**.
- [ ] **I7. Light/dark mode toggle**.
- [ ] **I8. Keyboard shortcuts**.
- [ ] **I9. Empty/loading/error states polish**.

### J. Documentation

- [ ] **J1. Hardware setup guide with photos**.
- [ ] **J2. Calibration guide with troubleshooting**.
- [ ] **J3. Cookbook / examples** — 5+ end-to-end recipes.
- [ ] **J4. Architecture overview with diagrams** — paper-quality.
- [ ] **J5. Contributing guide**.
- [ ] **J6. Auto-generated API reference**.

### K. Demo deliverables

- [ ] **K1. Demo script** — minute-by-minute live demo plan.
- [ ] **K2. Slide deck**.
- [ ] **K3. Recorded backup video**.
- [ ] **K4. README polish + badges**.
- [ ] **K5. Project landing page** at `farm.<domain>`.
- [ ] **K6. Final design doc revision** — incorporate what was actually built.

---

## Step 3: Explicitly out of scope

Cut entirely (vs Phase 2 deferred). Add only if specifically requested.

- **Multi-tenancy / auth / OAuth / Stripe billing** — not what this demo is about.
- **Planner / Dispatcher / Session DO split** — single Worker is fine for one user.
- **Multi-workspace concurrent execution** — single user means no concurrency design needed.
- **Bit-exact reproducibility across hardware** — physically not achievable with VLAs.
- **Real-time critic anomaly detection** — LLM latency makes <100ms infeasible.
- **Sim-twin per workspace (Mujoco/Isaac)** — interesting but a separate project.
- **Tactile / force-torque sensing** — RGB only.
- **Webhooks / event subscriptions** — polling is fine for a single user.
- **GR00T N1 as a third controller** — π0.5 + GPT-5 routing is enough.

---

## Step 4: How we work together

- **A, M, L, D, E, F, G, I, J** → orchestrator (parallel agents, PRs).
- **C (real robot), N (demo tuning)** → serial-with-gates (you validate each motion before I move on).
- I never push to `main` without a green PR.
- I run all tests before opening a PR.
- AI Gateway caching keeps OpenAI costs low.
- Cost meter surfaces per-skill spend in the UI.

---

## Step 5: Risk register

| Risk | P | Impact | Mitigation |
|---|---|---|---|
| xArm SDK API drift from shim | High | Med | C1 is day-1, before any motion. |
| Calibration eats half a day | High | Low | Accept 5mm tolerance for v1. |
| GPT-5V scene understanding flaky for novel scenes | High | Med | Cache descriptions; retry-with-better-photo loop. |
| Code-as-policy generates unsafe code | Med | High | AST-check, sandbox subprocess, safety enforcer is source of truth. |
| Skill library bloats with bad skills | Med | Med | N successful runs gate; eval harness validates. |
| LoRA training fails to improve on classical | Med | Low | Falls back to L2 (generated code). Per-layer success tracked. |
| Cube physical execution simply impossible | High | Low | Tier 4 is honest-failure-analysis by design. |
| First real-arm motion collides | Med | High | Tiny envelope, 0.05 m/s cap, e-stop in hand. |
| OpenAI costs balloon | Med | Low | AI Gateway cache + per-skill cost tracking. |

---

## Step 6: Day 0 — what to do right now

- [ ] Create `.env` with `OPENAI_API_KEY` (mandatory) and `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` (within a day or two)
- [ ] Confirm arm pings, cameras enumerate, e-stop accessible
- [ ] Tell me "go" — I start with A0 + A1
