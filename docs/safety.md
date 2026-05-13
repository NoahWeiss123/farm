# Safety

The Edge Agent is the safety boundary. No command reaches the arm without
passing every gate below. The gates are deterministic — they run in
constant time on the Edge Agent's process, never on the cloud. LLM-based
checks (the trailing critic) are explicitly not safety-critical; they
annotate run records, they do not stop the arm.

The gates are first-class events in the telemetry stream. They appear in
the dashboard alongside model thinking and trajectory data so a peer
reviewer can audit why the arm stopped.

## Workspace envelope

A configurable axis-aligned bounding box of allowed TCP poses, in the
arm's base frame.

- Configured at `safety.workspace_envelope` in
  [config.yaml](config-reference.md).
- Default: a conservative 40 cm cube centered in front of the arm.
- Any commanded pose outside the envelope is rejected before the move
  executes. The Edge Agent emits [FARM-E3001](errors.md#farm-e3001) and
  hands the run off to the next fallback in the chain.

To tune the envelope, write a YAML file with two corner poses:

```yaml
# envelope.yaml
min: {x: 0.10, y: -0.30, z: 0.05}    # base-frame metres
max: {x: 0.60, y:  0.30, z: 0.50}
```

Point the config at it:

```bash
farm config set safety.workspace_envelope ./envelope.yaml
```

A demo task should be designed so a planned overshoot trips the envelope
on purpose; the recovery is the most informative thing a reviewer can
watch.

## Velocity and acceleration caps

A hard ceiling on joint velocity and TCP linear velocity.

- Configured at `safety.velocity_cap_mps` (default 0.25 m/s).
- The Edge Agent rate-limits action chunks; any chunk requiring motion
  above the cap is clamped and the clamp is logged in the run record.
- Lower the cap for fragile objects or shared spaces:

  ```bash
  farm config set safety.velocity_cap_mps 0.10
  ```

## Singularity and self-collision check

Every commanded waypoint is run through the xArm SDK's IK and
self-collision check before execution. Failed checks pause the chunk and
the dispatcher invokes a recovery primitive (typically `relocalize` then
re-plan) before the next backend in the fallback chain takes over.

There is no user-tunable knob here; the check is structural.

## Watchdog

If the dispatcher's WebSocket goes silent for longer than the
`safety.watchdog_timeout_ms` window (default 1000 ms), the Edge Agent
halts the arm in place and emits [FARM-E3002](errors.md#farm-e3002). The
cloud cannot stall the arm in a dangerous configuration: the agent is
authoritative on this gate.

To tune for a flaky network, raise the timeout:

```bash
farm config set safety.watchdog_timeout_ms 2000
```

The watchdog also fires on local stalls (a chunk inference that exceeds
the agent's local 250 ms deadline reverts to halt). Do not set the value
below 250 ms.

## Physical e-stop

The UFactory 850's hardware e-stop is wired through the Edge Agent's
start sequence. The agent refuses to begin a run if the e-stop loop is
not detected. There is no software override; the gate is hardware.

[`farm doctor real-arm`](cli-reference.md#farm-doctor-real-arm) checks the
loop interactively before any run.

## Human-in-the-loop run gate

Every autonomous run requires a UI confirmation before execution starts.
Demo recordings happen with the operator in arm's reach. The gate
applies in Phase-MVP to every workspace; there is no opt-out.

## Calibration drift detection

The Edge Agent hashes `camera.<view>.intrinsics` at every run start and
embeds the hash in the [run record](python-api.md#run-records).

- If the calibration file's mtime is older than 24 hours, the run is
  refused with [FARM-E1002](errors.md#farm-e1002) unless
  `--accept-calibration` is passed.
- Run records with mismatched calibration hashes are grouped in the
  dashboard so peer reviewers can see which runs share a baseline.

Re-calibrate after any camera mount touch, even if it looked the same:

```bash
farm calibrate --camera wrist
```

## The trailing critic

A Critic LLM annotates the run record at each pause point (between
chunks, at end of plan node, at end of run). It is **not** a real-time
safety boundary — LLM call latency is too slow for that. By the time a
critic conclusion arrives, the next action chunk is already executing.

What the critic is good for: catching failure *patterns* across a run
for replay and learning, and as a passive monitor whose notes show up in
the dashboard's safety panel alongside the deterministic events.

What it is not: a substitute for any gate above.

## Recovery primitives

When the dispatcher walks the fallback chain on a backend failure, it
invokes one of these low-level primitives before the next backend takes
over. They are exposed in the [Python API](python-api.md) for power users
who want to compose them directly.

- `home` — move to a known safe pose, gripper open. Deterministic via the
  xArm SDK.
- `open_gripper` — release any grasped object. Used before retrying a
  failed grasp.
- `relocalize` — capture fresh frames from all cameras, run perception,
  return an updated `RunState`.
- `retry_grasp` — re-attempt the last grasp from the current TCP pose.
  For slip recovery, not for planning errors.
- `abort_safely` — descend to the nearest in-envelope waypoint, gripper
  open, watchdog disarmed. Terminal — the run ends.

The primitives are first-class capabilities, not opportunistic hacks.
Each backend's [capability card](capability-cards.md) declares whether it
can consume mid-run handoff state; backends that cannot get a
`home` + `relocalize` before the dispatch.

## Offline mode

```bash
farm run "wave hello" --offline
```

Routes everything through the local classical-planner backend. No cloud
calls. The same envelope, velocity, watchdog, e-stop, and calibration
gates apply. Use offline mode when:

- The network is down or unreliable
- You are testing a deterministic baseline against the same envelope
- A site policy forbids streaming camera frames

The capability ceiling is lower than the VLA path — the classical planner
cannot pick "the red one" without a color-detection bolt-on — but the
reliability floor is higher.
