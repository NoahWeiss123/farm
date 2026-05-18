# Safety

The Edge Agent is the safety boundary. No command reaches the arm without passing every gate below. Gates run on the Edge Agent process in constant time, never on the cloud.

The current build implements all gates but only the envelope, velocity, and watchdog gates are wired into the live run loop. Singularity, calibration, and e-stop modules are present, tested, and waiting for the real-arm path.

## Workspace envelope

Axis-aligned bounding box of allowed TCP poses in the arm's base frame.

- Configured at `safety.workspace_envelope` in [config.yaml](config-reference.md).
- Default: a conservative 40 cm cube centered in front of the arm.
- Any commanded pose outside the envelope is rejected before the move executes. The Edge Agent emits [FARM-E3001](errors.md#farm-e3001) and the run aborts.

```yaml
# envelope.yaml
min: {x: 0.10, y: -0.30, z: 0.05}    # base-frame metres
max: {x: 0.60, y:  0.30, z: 0.50}
```

## Velocity caps

Hard ceiling on TCP linear velocity.

- `safety.velocity_cap_mps` (default 0.25 m/s).
- The Edge Agent rate-limits action chunks. Chunks above the cap are clamped and the clamp is logged.

## Singularity and self-collision check

Every commanded waypoint runs through the driver's IK and self-collision check before execution. Failed checks halt the chunk. Not user-tunable.

## Watchdog

If the dispatcher's WebSocket goes silent for longer than `safety.watchdog_timeout_ms` (default 1000 ms), the Edge Agent halts the arm in place and emits [FARM-E3002](errors.md#farm-e3002). Do not set below 250 ms.

## Calibration drift

The Edge Agent hashes `camera.<view>.intrinsics` at run start. If the file's mtime is older than 24 hours, the run is refused with [FARM-E1002](errors.md#farm-e1002) unless `--accept-calibration` is passed.

## Physical e-stop

The UFactory 850 e-stop is wired through the Edge Agent's start sequence. The agent refuses to begin a run if the e-stop loop is not detected. Hardware gate, no software override. Not active in the sim path.

## Recovery primitives

When a backend fails, the dispatcher walks the fallback chain. Before the next backend takes over, one of these runs:

- `home` move to a known safe pose, gripper open.
- `open_gripper` release any grasped object.
- `relocalize` capture fresh frames, run perception, return updated state.
- `retry_grasp` re-attempt the last grasp from the current TCP pose.
- `abort_safely` descend to nearest in-envelope waypoint, gripper open. Terminal.

The primitives exist in `farm_edge_agent.recovery.primitives` and are tested. They're orphaned until the cloud dispatcher is wired up.
