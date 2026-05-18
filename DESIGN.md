# FARM design

Foundation for Action-Reasoning Models. A hosted agent harness that takes a natural-language task and runs it on a robot arm. CS153 final project.

This doc covers what FARM is, how it's structured, and the design decisions behind the layout. For the day-to-day "how do I run it" path, see `README.md`. For what's queued, see `IMPLEMENTATION_PLAN.md`.

## The product in one paragraph

A user types a task ("stack the red block on the cup"). The planner decomposes it into named skills using the live scene and the skill catalog. Each skill emits action chunks (TCP waypoints + gripper commands). Chunks pass through a safety enforcer (envelope, velocity, singularity, watchdog) before reaching the driver. The driver moves the arm (real UF850, or its MuJoCo twin). Every event is appended to a JSONL run record. Successful runs build up a plan cache, so repeat prompts skip the LLM.

## Why edge + cloud

The system splits along a cloud / local seam. Model inference, planning, and history live in the cloud. The real-time control loop and the arm live in a local Edge Agent. Pushing a 30 Hz action loop over WAN would add 100 to 300 ms RTT per step, which is fine for chunked actions but unworkable for closed-loop control.

The cloud serves the model. The Edge Agent runs the loop.

In the current local-only build both halves run in the same process. The seam is preserved so the cloud split is incremental refactoring, not a rewrite.

## Architecture (current build)

```
Browser dashboard (Next.js)
  POST /v1/runs                     SSE /v1/runs/:id/events
        |                                    ^
        v                                    |
  RunSupervisor (FIFO queue)
        |
        v
  RunLoop ── GptPlanner ── SkillExecutor ── SafetyEnforcer ── SimDriver (MuJoCo)
                |                                                  |
                v                                                  v
         plan_cache.db                                       EventBus + JSONL
```

The dashboard talks to a local aiohttp daemon (`farm serve`). The daemon owns a single SimDriver, a queue of pending runs, and an event bus. A worker thread pops one run at a time, constructs a RunLoop, and walks it to completion.

## Architecture (target, with cloud)

```
Browser dashboard ── Cloudflare Worker (planner) ── Dispatcher DO (per-run state)
                                                          |
                                                          | WebSocket
                                                          v
                                                    Edge Agent on the user's machine
                                                          |
                                                          v
                                                    Real UF850 or sim
```

The Worker handles planning. The Dispatcher DO owns one run's WebSocket session, walks the plan DAG, and multiplexes obs/action streams between Edge Agent and chosen backend. The Edge Agent stays the safety boundary regardless of which backend the cloud picked.

## Key concepts

### Capability cards

A capability card is the structured description the router reads when deciding which backend to use for a plan node. Versioned JSON Schema at `farm-shared/schemas/capability_card.v1.json`. The schema is the contract between backends and the router; changing it requires updating both the validator and consumers.

```yaml
id: pi05-ufactory-ft-v1
name: "π0.5 (fine-tuned on UFactory 850)"
roles: [controller]
embodiment: {arm: ufactory-850, dof: 6, action_space: ee_pose_delta_base_frame, control_rate_hz: 30}
skills:
  - pick:  {confidence: 0.80, learned_from: 240_demos}
  - place: {confidence: 0.85, learned_from: 240_demos}
  - stack: {confidence: 0.70, learned_from: 60_demos}
latency: {p50_ms_per_chunk: 95, p99_ms_per_chunk: 220}
cost_per_chunk_usd: 0.0008
determinism: stochastic
safety: {requires_envelope: true, supports_velocity_cap: true}
fallbacks: [classical-planner-pick, gemini-robotics-act]
```

Cards are embedded in each run record so old records stay interpretable as the schema evolves.

### Plan DAG with fallback chains

The planner returns a DAG. Each node references a skill and carries an explicit ordered fallback list. The dispatcher walks the list on any of: HTTP error, timeout, safety stop, or critic-flagged deviation. Default order is primary backend, then next-best capability match, then the classical planner running in the Edge Agent.

Two hard limits prevent infinite-loop pathologies: `max_attempts_per_node` (default 2) and `max_wall_clock_per_run` (default 5 min). Both surface in the run record.

### State handoff at fallback boundaries

When the dispatcher walks to the next backend, the new backend cannot pick up "wherever the last one left off" without an explicit handoff. The Edge Agent halts at the next chunk boundary, reports its RunState (joint pose, TCP pose, gripper state, task-progress index, last completed chunk, fresh observation, critic summary), and a recovery primitive (`home`, `open_gripper`, `relocalize`, or `abort_safely`) runs before the new backend takes over.

Backends that can't consume partial-progress state declare this in their capability card; the dispatcher then invokes `home` + `relocalize` before handing off.

### Safety boundary

The Edge Agent is the safety boundary. No command reaches the arm without passing every gate. Gates are deterministic, constant-time, and run locally. LLM-based checks (the trailing critic) annotate run records but never stop the arm. See `docs/safety.md` for the full list.

### Run records

Append-only JSONL, one file per run at `~/.farm/runs/<id>/record.jsonl`. Replay must be deterministic from the record alone. Records carry the original prompt, the plan DAG with router reasoning per node, every action chunk, downsampled observations, every safety event, and the wall-clock + cost breakdown.

HMAC signing against a workspace key is planned. `farm verify <run-id>` will check signature + lock match. This is a research-grade audit trail (the verifier holds the key), not manufacturing-grade compliance.

### Protocol versioning

Every Edge Agent ↔ Dispatcher WebSocket handshake exchanges `protocol_version` (semver, bumped on breaking changes), `agent_version` (informational), and `feature_flags`. Mismatched protocol version causes the dispatcher to reject the connection with [FARM-E1006](docs/errors.md#farm-e1006).

## Backend types

Backends split into three roles. A single model can fill more than one.

```python
class Planner(Protocol):
    capability_card: CapabilityCard
    async def plan(self, task: str, scene: Observation) -> PlanDAG: ...

class Controller(Protocol):
    capability_card: CapabilityCard
    async def act(self, obs_stream, instruction: str) -> AsyncIterator[ActionChunk]: ...

class Critic(Protocol):
    capability_card: CapabilityCard
    async def critique(self, trajectory: Trajectory) -> CriticReport: ...
```

Current build: GPT planner (OpenAI), Python skill library as a controller, classical fallbacks for each skill. Critic is not wired up.

## Latency budget

- Plan generation: 2 to 5 s, once per task. Plan cache eliminates this on repeat prompts.
- Inference per action chunk: 80 to 200 ms for a hosted VLA, ~free for the classical path.
- Network RTT cloud to Edge Agent: 30 to 100 ms (PoP-dependent).
- Local control loop: 33 ms cadence (30 Hz), driven by the chunk buffer.
- End-to-end, prompt to arm starts moving: 3 to 6 s warm, 8 to 25 s cold.

Chunked actions (π0-FAST emits multi-step sequences per inference) are what make cloud serving viable. Without them, this architecture wouldn't work.

## Reproducibility

A workspace can commit a `farm.lock` file pinning the Edge Agent version, protocol version, backend digests, and calibration hash. `farm verify <run-id>` checks the run was executed against those locks. Useful for paper appendices and experiment scripts.

Not in scope: third-party-verifiable signing (would need asymmetric keys), proof the arm actually executed the logged trajectory (would need hardware-side telemetry), operator identity, e-stop state, PLC state.

## Key invariants

- The capability-card schema is the contract between backends and the router. Changing it requires updating both the validator and consumers.
- `ErrorCode` is an Enum of `_Spec` dataclasses. Always render with `format_error(code, **slots)`. Never f-string the enum.
- Two `FarmError` classes exist on purpose: `farm_edge_agent.errors.FarmError` (structured) and `farm_edge_agent.client.FarmError` (simple Exception for the public Python API). Don't merge them.
- Run records are append-only JSONL. Replay must be deterministic from the record. Don't write timestamps as source of truth.
- Tests are deterministic. No `sleep()`, no unseeded randomness. Network/hardware tests are in `*_integration_test.py` files.

## What's deferred

- Real-arm path. The xArm driver shim exists, `farm doctor real-arm` is stubbed. Needs SDK verification, hand-eye calibration, and the safety envelope measured against actual workspace geometry.
- Vision perception. Object positions come from sim ground truth. The affordance reasoner is replaced by the static skill catalog.
- Layer-3 LoRA skill compiler. Plan cache (Layer 1) is live. Parameterized code (Layer 2) is implicit in the skill library. The LoRA fine-tune pipeline is a future Modal job.
- Multi-tenant auth, billing, observability. Out of scope for the demo.

See `IMPLEMENTATION_PLAN.md` for the status grid and the queued work.
