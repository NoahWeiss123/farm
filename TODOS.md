# FARM — Deferred Work

Items considered during design. Two flavors: **promoted into Phase-MVP** (items 4 and 5 below — listed for traceability; the actual work landed in DESIGN.md) and **deferred to Phase-Product or backlog** (items 1, 2, 3, and everything below the divider).

A TODO without context decays into noise. Don't add a one-liner. Write the next person (probably future-you) enough context to act on it.

---

## Phase-Product (post-CS153)

### 1. Planner / Dispatcher / Session split

**What.** Refactor the Phase-MVP single-Worker harness into three components: a stateless Planner Worker (LLM tool-call routing), a Dispatcher Durable Object (per-run state + telemetry fan-out), and a Session DO (history, replay, run-record persistence).

**Why.** Phase-MVP collapses these into one Worker for delivery speed. The split matters when (a) concurrent runs per workspace need isolation, (b) WebSocket sessions outlive a Worker's 30s CPU limit, (c) Planner becomes its own scaling unit because LLM calls dominate latency. None of that is true with one user; all of it will be true with twenty.

**Pros.** Clean scaling boundaries. Independent deployability. Each component has one responsibility.
**Cons.** Three deploy targets. Three places state can drift. Worth nothing until concurrent users exist.

**Context.** The Architecture section of DESIGN.md describes the *target* shape. The Phase-MVP code should use interfaces shaped like the Phase-Product split (e.g., a `planner.plan(task, scene)` method, even if it's a function call inside one Worker), so the eventual split is mechanical, not a rewrite.

**Depends on.** Phase-MVP shipped. Second user signed up.

---

### 2. Multi-workspace concurrent execution

**What.** Allow N concurrent runs per workspace and across workspaces, with GPU multiplexing on the π0.5 inference container and per-workspace quota enforcement on the Dispatcher.

**Why.** Phase-MVP runs you, alone. Phase-Product needs to demonstrate that the harness scales beyond one demo. The hardest part is GPU multiplexing — π0.5 is large enough that running two concurrent inferences on one L4 either OOMs or 4x's latency.

**Pros.** Unlocks the actual hosted-inference business story. Required for the manufacturing-pilot conversation.
**Cons.** Real concurrency means real race-condition surface area. Load-testing this is its own project.

**Context.** AI Gateway already does cost attribution per workspace. Quotas need to be enforced at the Dispatcher (workspace-scoped rate limit) and the Container (per-request priority). R2 prefixes are already workspace-scoped in Phase-MVP, so storage isolation is free.

**Depends on.** Phase-Product Planner/Dispatcher/Session split (TODO #1).

---

### 3. Bring-your-own-model registration API

**What.** A public HTTP endpoint where external developers POST a capability card pointing at their own model server. The FARM router considers the registered model alongside hosted backends.

**Why.** This is the ecosystem move. It turns FARM from "a product" into "a marketplace." Cheap to implement (the capability-card format is already the contract); huge story for the research-lab and startup personas.

**Pros.** Ecosystem flywheel. Zero hosting cost on FARM's side for BYO models. Differentiates against LeRobot's library-only posture.
**Cons.** Trust boundary: a malicious capability card could lie about its capabilities to manipulate routing. Need an attestation or rating system before this can be safely public.

**Context.** Capability-card YAML schema is fixed in Phase-MVP. The router treats cards as tool definitions; an externally-hosted card is mechanically identical. The trust problem is the gating issue, not the engineering.

**Depends on.** Stable capability-card schema (Phase-MVP). A trust/attestation design.

---

### 4. Calibration drift detection in run records *(Phase-MVP)*

**What.** Edge Agent computes a hash of the camera intrinsics + extrinsics file at run start and writes it into the run record. Refuses to start a run if the calibration file mtime is older than 24h without explicit `--accept-calibration` flag. UI surfaces calibration-hash grouping so peer reviewers can see which runs share a baseline.

**Why.** Camera mount bumps are the #1 silent failure mode in lab-grade robotics projects. Without this, a run can silently regress days after a productive collection session, and the cause is invisible.

**Pros.** One day of work. Eliminates the most common "why did the model suddenly get worse" debugging nightmare. Required for the research-lab persona's reproducibility claim to be real.
**Cons.** None worth listing — this is the kind of feature that has to ship.

**Context.** Hash format: SHA-256 of `intrinsics.yaml || extrinsics.yaml` bytes. Surface as `calibration_hash` in the run record schema. UI: group runs by hash, color-code, show "calibration changed between run N and run N+1" markers.

**Depends on.** Run record schema (Phase-MVP).

**Status.** Promoted into Phase-MVP per eng review. Now part of Safety section in DESIGN.md. Listed here for traceability.

---

### 5. Version negotiation protocol Edge Agent ↔ Dispatcher *(Phase-MVP)*

**What.** WebSocket handshake exchanges `protocol_version` (semver, wire-schema versioned) and `agent_version` (informational). Mismatched protocol → structured error + upgrade instructions. Optional `--auto-update` flag pip-upgrades the Edge Agent and reconnects.

**Why.** Edge Agent is pip-installed, Cloud ships on its own cadence, drift is inevitable. Without negotiation, drift manifests as silent action-schema mismatches and nonsense arm motion. No user can debug that without source access.

**Pros.** Half a day of work. Prevents an entire class of "the demo broke and I can't figure out why" failures. Mandatory for the public sandbox to be usable by reviewers.
**Cons.** None.

**Context.** Use semantic versioning of the wire protocol, decoupled from package versions. Bump major when chunk schema or control messages change. Edge Agent CLI surfaces the upgrade command verbatim. Auto-update is opt-in for manufacturing users (who care about stability), opt-out for sandbox users (who don't).

**Depends on.** WebSocket handshake exists (Phase-MVP).

**Status.** Promoted into Phase-MVP per eng review. Now part of Architecture section in DESIGN.md. Listed here for traceability.

---

## Backlog (uncategorized, revisit each)

### Bit-exact reproducibility across hardware

Out of scope. Flow-based VLAs + heterogeneous GPU floating-point makes this physically not achievable without bespoke kernels. Honest claim is "same plan DAG, deterministic classical legs, seeded-stochastic VLA legs" — this lives in DESIGN.md as **Reproducible Mode**. Don't promise bit-exactness in marketing.

### Real-time critic anomaly detection

Out of scope. LLM call latency makes real-time critic impossible. Safety is the Edge Agent's deterministic checks. Critic produces trailing annotations only. If a future cheap+fast vision model makes <100ms anomaly detection feasible, revisit.

### A docs site (rendered, hosted, with search)

Phase-Product. Phase-MVP ships `docs/` markdown in the repo + GitHub Pages auto-render. A real docs site (algolia search, versioning, runnable code samples) is post-CS153.

### GR00T N1 as a third controller

Stretch goal, listed in DESIGN.md. Adds variety to the auto-mode story. Defer until π0.5 + Gemini + classical is reliably working — three backends is enough to validate the routing thesis, four is for polish.

### Sim-twin per workspace (Mujoco / Isaac Sim mirror)

Stretch goal. Researchers will pay for this alone. Real engineering: a second backend that takes the same action chunks and feeds them into a sim, with the UI showing both side-by-side. Estimate: 1-2 weeks once the action-chunk plumbing is stable.

### CSV / Notebook export of run records

Stretch goal. The run record format is already structured; this is `pandas.DataFrame.from_records(...)` and a one-button download. Worth doing if a research-lab early adopter asks.

### Algolia-indexed docs site

Phase-Product. Phase-MVP ships `docs/` markdown auto-rendered by GitHub Pages. A real docs experience with versioning, search, runnable code samples, and inline API playground is post-CS153. Plan for it: keep markdown structure compatible with Docusaurus or VitePress so the migration is mechanical.

### Run-record signing infrastructure (HMAC → Cosign)

Phase-MVP ships HMAC signing of run records using a workspace signing key. This is enough for `farm verify` to detect tampering by anyone outside the workspace. **Phase-Product upgrade**: move to Cosign-style asymmetric signing with the public key published, so external auditors (compliance, regulators, paper reviewers) can verify records without trusting FARM. Triggers: first manufacturing pilot conversation that mentions compliance.

### Multi-camera depth + tactile sensing

Out of scope for Phase-MVP (wrist + optional overhead RGB only). Some manufacturing tasks (insertion, deformable manipulation) need force/torque or tactile feedback that the current perception layer does not capture. Capability card schema is forward-compatible — `input_modalities` can grow — but no backend in Phase-MVP consumes these. Revisit when a controller backend that takes tactile input is available off-the-shelf.

### Webhooks + event subscriptions

Phase-Product. Research-lab integration story currently requires polling the run record. A webhook surface (`run.started`, `run.completed`, `safety_event.fired`, `calibration.drift_detected`) lets external systems integrate without polling. Cheap to add once the Dispatcher DO is a real Durable Object (it already fan-outs internally); not worth doing before the Planner/Dispatcher/Session split.
