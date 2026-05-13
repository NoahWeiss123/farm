# FARM error catalog

Every error the Edge Agent surfaces is `[FARM-Exxxx] <one-line> — fix: <action>`.
Each entry below mirrors what the CLI prints, with a longer description, the
common causes, and fix steps. Canonical online location is
`https://farm.dev/errors/E<NNNN>`.

## FARM-E1001

The Edge Agent could not open the configured camera device.

**Common causes**

- The camera is unplugged or powered off.
- The OS reassigned the device node (`/dev/video0` → `/dev/video2`) after a
  reboot or hotplug.
- Another process (a browser tab, OBS, a stale Edge Agent) has the device
  open exclusively.
- The user running `farm` is not in the `video` group on Linux.

**Fix**

1. Run `farm doctor cameras` to list every device the OS exposes and what
   the Edge Agent sees.
2. If the device moved, point the config at the new node:
   `farm config set camera.wrist.device /dev/videoN`.
3. If a stale process holds the camera, kill it
   (`lsof /dev/video0` on Linux, `ioreg` on macOS).

Docs: <https://farm.dev/errors/E1001>

## FARM-E1002

The on-disk calibration is older than the freshness window the run requires.

**Common causes**

- The arm or camera has been re-mounted since the last calibration.
- The workspace policy requires calibration every N days; the cadence elapsed.
- A new end effector was swapped in without re-running calibration.

**Fix**

1. Re-run `farm calibrate` end-to-end (camera intrinsics, hand–eye, gripper).
2. Or, if you accept the risk for this single run, pass
   `--accept-calibration` to bypass the check.

Docs: <https://farm.dev/errors/E1002>

## FARM-E1003

The cloud GPU container that serves the configured backend is cold-starting.
This is not a failure; the run is held open and will resume the moment the
container is warm.

**Common causes**

- First request to a workspace's GPU pool after an idle period.
- The backend image was redeployed and the warm pool was drained.
- A burst of concurrent runs scaled out a new replica.

**Fix**

1. Wait. Typical cold-start is 8–25 s; the CLI shows a live elapsed counter.
2. If cold-starts dominate your workflow, consider switching the workspace
   to a backend with a pinned warm pool (Phase-Product feature).

Docs: <https://farm.dev/errors/E1003>

## FARM-E1004

The Dispatcher rejected the API key supplied by this Edge Agent.

**Common causes**

- `FARM_API_KEY` is unset, empty, or contains stray whitespace.
- The key was rotated or revoked from the workspace settings.
- The Edge Agent is pointed at a different workspace than the one the key
  was issued for.

**Fix**

1. Re-authenticate with `farm login` and follow the device-code flow.
2. Or verify `echo $FARM_API_KEY` matches what the workspace page shows.

Docs: <https://farm.dev/errors/E1004>

## FARM-E1005

The Dispatcher WebSocket connection dropped mid-run. The arm halts in place
and the Edge Agent attempts to auto-reconnect.

**Common causes**

- Transient network glitch (WiFi roam, captive portal re-auth, VPN flap).
- The Dispatcher rolled to a new revision and closed open sockets.
- A middlebox idle-timed the WebSocket out.

**Fix**

1. Wait for the auto-reconnect; the CLI shows progress.
2. If reconnect does not restore the run, resume from where it stopped:
   `farm run --resume <run-id>`.
3. If the network is unreliable, try `FARM_RELAY=on` to tunnel over HTTPS
   long-polling.

Docs: <https://farm.dev/errors/E1005>

## FARM-E1006

The Edge Agent's protocol version is older than the Dispatcher requires.

**Common causes**

- The Edge Agent was installed months ago and never upgraded.
- The Dispatcher was rolled forward to a new protocol minor version.
- The user is on a fork or pre-release build that lags the public protocol.

**Fix**

1. Upgrade in place: `pip install -U farm-edge-agent`.
2. Or pass `--auto-update` once so future runs upgrade themselves on mismatch.

Docs: <https://farm.dev/errors/E1006>

## FARM-E1007

The network probe could not complete a WebSocket upgrade to the Dispatcher.
The session cannot start until a transport works.

**Common causes**

- A corporate proxy strips the `Upgrade: websocket` header.
- A captive portal is intercepting outbound TLS.
- A firewall rule blocks outbound 443 to the Dispatcher hostname.

**Fix**

1. Run `farm doctor network` for a layered DNS → TLS → WS → RTT diagnosis.
2. Set `FARM_RELAY=on` to fall back to HTTPS long-polling.
3. Or run `farm run --offline` to use the local classical-planner backend.

Docs: <https://farm.dev/errors/E1007>

## FARM-E2001

A capability card's `action_space` field is not in the schema's allowed set.

**Common causes**

- A typo in the YAML (`ee_pose_delta` vs `ee-pose-delta`).
- The card was written against an older schema version with a different
  enum.
- A new action space was proposed but not yet shipped in the schema.

**Fix**

1. Inspect the suggestion the validator printed and update the YAML to
   match.
2. Cross-check against the published schema at
   <https://farm.dev/schemas/capability_card.v1>.
3. Re-run `farm card validate <file>` until it passes before submitting
   the card to a run.

Docs: <https://farm.dev/errors/E2001>

## FARM-E3001

The Edge Agent's safety layer rejected a commanded pose because it falls
outside the configured workspace envelope. The arm is soft-stopped in
place; no motion was executed.

**Common causes**

- The model produced a trajectory that drifts outside the envelope.
- The envelope is mis-configured (too small, wrong frame, stale after a
  base move).
- A coordinate-frame mismatch between the camera and the arm shifted the
  whole plan.

**Fix**

1. Inspect the rejected pose in the run record and compare with the
   envelope in `~/.farm/config.yaml`.
2. If the envelope is wrong, widen or re-center it; do not disable it.
3. If the model is consistently producing out-of-envelope plans, swap
   backends or fall back to the classical planner.

Docs: <https://farm.dev/errors/E3001>

## FARM-E3002

The Edge Agent's watchdog tripped because the Dispatcher went silent for
longer than the 1-second budget. The arm is halted in place.

**Common causes**

- The Dispatcher process is slow or stuck on this run.
- A network event blocked send-side throughput for >1 s.
- The local machine is starved for CPU (background job, swap thrash).

**Fix**

1. Check Dispatcher status on the workspace dashboard.
2. Run `farm doctor network` to look for latency spikes.
3. If the local box is overloaded, free up resources or move the Edge
   Agent to a dedicated host.

Docs: <https://farm.dev/errors/E3002>
