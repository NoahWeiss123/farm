# FAQ

The questions that get asked first. If yours isn't here and the
[errors.md](errors.md) page doesn't cover it, open an issue with the run
ID and the output of `farm doctor`.

## Why is my first run slow?

The cloud inference container has to spin up on a cold workspace. The
typical cold-start window is **8–25 seconds**; subsequent runs are
sub-second. The Edge Agent emits
[FARM-E1003](errors.md#farm-e1003) for the duration so the dashboard can
show a "warming up" panel instead of a featureless spinner.

What you can do:

- The quickstart sandbox is pre-warmed for the first 60 seconds after
  `farm quickstart`. If you came in through that path, your first prompt
  will not hit a cold start.
- For your own workspace, pre-warm by running any cheap task before your
  demo:

  ```bash
  farm run "wave hello" --backend classical
  ```

  The classical run does not load the GPU container, but the dispatcher
  warms its caches and the connection stays alive long enough that the
  next prompt usually lands warm.
- Cold start does not affect `--offline` runs.

## My WebSocket won't connect

Symptom: [FARM-E1007](errors.md#farm-e1007) at startup, or the connection
flaps every few seconds.

First, run the network probe:

```bash
farm doctor network
```

The probe walks DNS → WebSocket upgrade → RTT → TLS chain → MTU →
throughput, prints a verdict per leg, and gives you a fix command. The
common culprits, in order:

1. **Captive portal** (coffee shop, conference, hotel). Sign in through
   a browser first; the WebSocket cannot answer the captive portal's
   redirect.
2. **Corporate proxy stripping the `Upgrade` header.** Run with
   `FARM_RELAY=on`:

   ```bash
   FARM_RELAY=on farm start
   ```

   The relay tunnels the WebSocket over HTTPS long-polling. Latency goes
   up; throughput drops slightly; behaviour is otherwise identical.
3. **VPN with low MTU.** Disconnect the VPN or raise the MTU. The probe
   prints the discovered MTU; anything below 1280 will hurt.
4. **Aggressive firewall** that closes long-lived connections. Pair the
   relay with a more permissive egress policy; long-polling looks like
   normal HTTPS.

## My calibration just changed and my runs got worse

That is the silent failure mode the calibration-drift detector exists to
catch. A camera mount that drifts a few millimetres between calibration
and inference produces noticeably worse VLA execution; the model is
trained on a precise extrinsic and is unforgiving when the world shifts
under it.

Symptoms:

- Sudden drop in success rate with no other change.
- The dashboard shows runs grouped under a new calibration hash.
- A run record carries [FARM-E1002](errors.md#farm-e1002) at start.

Fix order:

1. Inspect the camera mount. Look for screws backed off, padding
   compressed, or a tool flange that was bumped.
2. Re-calibrate:

   ```bash
   farm calibrate --camera wrist
   ```

3. Re-run the failing task. If success returns immediately, you found
   the cause. If it does not, see the
   [training runbook](training-runbook.md).

The dashboard's run-history view groups runs by calibration hash, so a
"calibration changed between run N and run N+1" marker tells you when
the baseline shifted.

## How do I run without an internet connection?

Use offline mode:

```bash
farm run "pick the red block" --offline
```

Everything runs through the local classical-planner backend; no cloud
calls. Capability ceiling is lower than the VLA path, but the gates in
[safety.md](safety.md) all still apply. Use this for development on a
plane, for sites with no egress, or to test a deterministic baseline.

## How do I share a run with a collaborator?

```bash
farm export r_8x2k --format jsonl --out ./r_8x2k.jsonl
```

The JSONL file contains the prompt, plan DAG, action chunks,
observations (downsampled), safety events, and critic notes. For a
LeRobot-shaped trajectory shard, use `--format lerobot`. See
[python-api.md](python-api.md#exporters).

## Why does `farm doctor` warn about my calibration when I just ran it?

The 24-hour window starts at file mtime. If the file was copied or
restored from a backup, its mtime may reflect the original creation
time, not your most recent re-calibration. Either re-run `farm
calibrate` to write a fresh file or pass `--accept-calibration` when you
know the calibration is current.

## What does Phase-MVP versus Phase-Product mean in the docs?

FARM ships in two phases. **Phase-MVP** is what shipped for CS153; it is
the surface this docs site describes. **Phase-Product** is the post-course
continuation — multi-tenant workspaces, the Planner/Dispatcher/Session
split, the bring-your-own-model API. Features marked Phase-Product in the
text are deliberately not in Phase-MVP. See `DESIGN.md` and `TODOS.md` for
the full scope split.

## My WebSocket says "protocol mismatch"

Wire protocol mismatch. The Edge Agent and the dispatcher each declare a
supported range; the dispatcher refuses the connection if there is no
overlap. See [upgrading.md](upgrading.md) for the protocol version table
and [FARM-E1006](errors.md#farm-e1006).

## My fine-tune sim eval is below 50%

That is the rollback threshold. Don't ship the checkpoint; diagnose
instead. The decision tree is in
[training-runbook.md](training-runbook.md).
