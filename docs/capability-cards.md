# Capability cards

A capability card is the structured description the LLM router reads when it
decides which backend to dispatch a plan node to. Cards are versioned as
JSON Schema, shipped with the Edge Agent at
`farm_edge_agent/schemas/capability_card.v1.json`, and embedded into each
[run record](python-api.md#run-records) so old records stay interpretable
when the schema evolves.

In Phase-MVP the set of cards is hard-coded — three backends, three cards:
fine-tuned π0.5, Gemini Robotics proxy, and the classical planner. The
[bring-your-own-model registration API](../TODOS.md) that lets external
developers register a card is Phase-Product.

## Worked example

```yaml
id: pi05-ufactory-ft-v1
name: "π0.5 (fine-tuned on UFactory 850)"
roles: [controller]
embodiment:
  arm: ufactory-850
  dof: 6
  action_space: ee_pose_delta_base_frame
  control_rate_hz: 30
input_modalities: [rgb_image, joint_state, language]
camera_views: [wrist, overhead_optional]
skills:
  - pick:  {confidence: 0.80, learned_from: 240_demos}
  - place: {confidence: 0.85, learned_from: 240_demos}
  - stack: {confidence: 0.70, learned_from: 60_demos}
  - pour:  {confidence: 0.40, learned_from: 0_demos}
latency:
  p50_ms_per_chunk: 95
  p99_ms_per_chunk: 220
cost_per_chunk_usd: 0.0008
determinism: stochastic
safety: {requires_envelope: true, supports_velocity_cap: true}
fallbacks: [classical-planner-pick, gemini-robotics-act]
```

## Required fields

- **`id`** — string, kebab-case. Stable identifier; appears in run records
  and in `--backend` flags on [`farm run`](cli-reference.md#farm-run).
- **`name`** — string, human readable. Surfaced in the dashboard.
- **`roles`** — list, one or more of `planner`, `controller`, `critic`.
  A single card may fill multiple roles; the architecture's three protocols
  (`Planner`, `Controller`, `Critic`) are not mutually exclusive.
- **`embodiment`** — object describing the arm assumption:
  - `arm`: one of the arm names in [hardware.md](hardware.md).
  - `dof`: integer.
  - `action_space`: enum. Phase-MVP value: `ee_pose_delta_base_frame`.
  - `control_rate_hz`: integer. Effective rate the backend supports; chunks
    are buffered to this rate at the Edge Agent.
- **`input_modalities`** — list, subset of
  `rgb_image`, `joint_state`, `language`, `depth_image`, `force_torque`.
  Phase-MVP supports only the first three.
- **`camera_views`** — list, subset of `wrist`, `overhead_optional`.
- **`latency`** — object with `p50_ms_per_chunk` and `p99_ms_per_chunk`.
- **`cost_per_chunk_usd`** — float. Used by the router for cost-weighted
  decisions.
- **`determinism`** — enum: `deterministic`, `stochastic`, `seeded`.
- **`safety`** — object with `requires_envelope` and
  `supports_velocity_cap` booleans. See [safety.md](safety.md).
- **`fallbacks`** — ordered list of backend ids the dispatcher walks if
  this backend fails. The chain must terminate at the classical planner;
  that is the architectural commitment that makes graceful recovery work.

## Optional fields

- **`skills`** — list of per-skill confidence + learned-from-demo
  annotations. The router reads `confidence` as a soft prior; a skill with
  `confidence < 0.5` is avoided unless it is the only option.
- **`description`** — short free-form text, used in the router prompt.

See `farm_edge_agent/schemas/capability_card.v1.json` for the authoritative
schema.

## Validating a card

```bash
farm card validate ./mycard.yaml
```

Errors are emitted in the structured form described in
[errors.md](errors.md#farm-e2001): JSON-pointer-style path, message,
suggested allowed value where applicable.

Programmatically:

```python
from farm import card

result = card.validate_file("./mycard.yaml")
for err in result.errors:
    print(err.path, err.message, err.suggestion)
```

## Skill names

`skills.<name>` is not enumerated — backends self-declare. The
router reads the name verbatim. The four names used in Phase-MVP eval are
`pick`, `place`, `stack`, `pour`; the classical-planner card lists `pick`,
`place`, `home`, `open_gripper`. Adding a skill is just a new key on a
card; no schema change required.

## Determinism and reproducibility

`determinism: deterministic` is the classical planner: same inputs produce
the same outputs, bit-exact, on the same machine.

`determinism: seeded` means the backend respects a `seed` field in the run
request and produces approximately the same trajectory across runs.
Cross-hardware bit-exactness is explicitly not promised — flow-based VLAs
on different GPUs drift.

`determinism: stochastic` makes no reproducibility promise. The router
prefers seeded over stochastic when the prompt explicitly requests
reproducibility.

See `farm.lock` and the Reproducibility section of DESIGN.md for the
full story, plus [upgrading.md](upgrading.md#lock-files-and-protocol-versions).
