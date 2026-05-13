# Errors

Every error emitted by `farm-edge-agent` follows the same shape:

```
[FARM-Exxxx] <one-line description> — fix: <command or action>
```

The structured form is reproduced in JSON output (`--json`) and in
[run records](python-api.md#run-records). Each code has a long-form page at
`https://farm.dev/errors/E<NNNN>`; the canonical text and fix steps are in
this file.

Codes are stable across the Phase-MVP wire protocol. New failure modes get
new codes; existing codes are not renumbered. See
[upgrading.md](upgrading.md) for protocol-version compatibility.

## Quick reference

| Code | Surface | One-liner |
|---|---|---|
| [FARM-E1001](#farm-e1001) | CLI | Camera device not found |
| [FARM-E1002](#farm-e1002) | CLI | Calibration is stale |
| [FARM-E1003](#farm-e1003) | CLI + UI | GPU container cold-starting |
| [FARM-E1004](#farm-e1004) | CLI | API key rejected |
| [FARM-E1005](#farm-e1005) | CLI + UI | Dispatcher WebSocket dropped |
| [FARM-E1006](#farm-e1006) | CLI | Wire protocol version mismatch |
| [FARM-E1007](#farm-e1007) | CLI | Network probe failed |
| [FARM-E1008](#farm-e1008) | CLI | Driver requires `arm.ip` in config |
| [FARM-E1009](#farm-e1009) | CLI | Config file not found |
| [FARM-E1010](#farm-e1010) | CLI | Required env var missing |
| [FARM-E2001](#farm-e2001) | CLI | Capability card validation failed |
| [FARM-E3001](#farm-e3001) | Edge Agent | Safety envelope violation |
| [FARM-E3002](#farm-e3002) | Edge Agent | Watchdog timeout |

## FARM-E1001

> No camera found at `<device>`. fix: `farm doctor cameras`, then `farm config set camera.wrist.device /dev/videoN`

The configured `camera.<view>.device` path does not enumerate as a video
device. Most often this is a `/dev/video0` vs `/dev/video2` mistake after a
USB hub re-enumerated. Run [`farm doctor cameras`](cli-reference.md#farm-doctor-cameras)
to see the actual list, then update the config.

Raised as `farm.errors.CameraNotFound` from the Python API.

## FARM-E1002

> Calibration is `<N>` days old. fix: `farm calibrate`, or pass `--accept-calibration`

The Edge Agent hashes `camera.<view>.intrinsics` at run start. If the file
is older than 24 hours, the run is refused unless `--accept-calibration` is
passed on the CLI. Camera-mount bumps are the #1 silent failure mode in
lab-grade robotics work; this gate makes the regression loud. See
[safety.md](safety.md#calibration-drift-detection).

## FARM-E1003

> GPU container cold-starting (typical 8–25s). Holding the run open; arm will move when ready.

Informational, not an error: the cloud inference container had to spin up.
First-prompt cold-starts are bounded; warm runs are sub-second. The
quickstart sandbox stays pre-warmed for the first 60 seconds after
`farm quickstart`. See [faq.md](faq.md#why-is-my-first-run-slow).

## FARM-E1004

> API key rejected. fix: `farm login`, or check `FARM_API_KEY` env var

The dispatcher rejected the workspace key during handshake. Causes:

- The key was rotated and the local copy is stale → re-run `farm login`.
- `${FARM_API_KEY}` resolves to an empty string → check the env var.
- The key belongs to a workspace that no longer exists.

## FARM-E1005

> Dispatcher WebSocket dropped after `<N>` s. Auto-reconnecting; arm halted in place. fix: `farm run --resume <run-id>`

The Edge Agent lost its connection mid-run. The watchdog halted the arm in
place; the dispatcher persisted state up to the last completed chunk. Use
[`farm run --resume`](cli-reference.md#farm-run) to continue from there.

## FARM-E1006

> Edge Agent v`<X>` detected, Dispatcher requires v`<Y>`+. fix: `pip install -U farm-edge-agent`

Wire-protocol version mismatch at handshake. The dispatcher refuses the
connection rather than risking silent action-schema drift. See
[upgrading.md](upgrading.md) for the protocol version table.

`farm start --auto-update` upgrades the package and reconnects automatically;
off by default for manufacturing users.

## FARM-E1007

> Network probe FAILED: WebSocket upgrade blocked. fix: `farm doctor network` for diagnostics; try `FARM_RELAY=on`

DNS resolved but the WebSocket upgrade was blocked, typically by a captive
portal, corporate proxy, or aggressive firewall.
[`farm doctor network`](cli-reference.md#farm-doctor-network) emits a
per-failure fix command. `FARM_RELAY=on` tunnels over HTTPS long-polling at
the cost of higher latency. See [faq.md](faq.md#my-websocket-wont-connect).

## FARM-E2001

> `capability_card.<path>`: `'<value>'` not in allowed set. Did you mean `'<suggestion>'`? Schema: `https://farm.dev/schemas/capability_card.v1`

A [capability card](capability-cards.md) failed schema validation. Enum
mismatches include a `Did you mean` suggestion based on edit distance.
Common causes: typo in `action_space`, missing `embodiment.arm`, unknown
value in `roles`.

## FARM-E3001

> Safety envelope violation: commanded pose outside workspace. Soft-stopped.

A planned chunk would have put the TCP outside the configured
`safety.workspace_envelope`. The Edge Agent soft-stopped the arm before the
move executed. The run continues with the next fallback in the chain. See
[safety.md](safety.md#workspace-envelope).

## FARM-E3002

> Watchdog timeout (>1s server silence). Arm halted in place.

The dispatcher's WebSocket went silent for longer than
`safety.watchdog_timeout_ms` (default 1000ms). The Edge Agent halted the arm
in place. The cloud cannot stall the arm in a dangerous configuration. See
[safety.md](safety.md#watchdog) and [FARM-E1005](#farm-e1005).

## FARM-E1008

> Driver `'<driver>'` requires `arm.ip` in config. fix: `farm config set arm.ip <robot-ip>`

The configured driver (e.g. `xarm`) needs a network address for the arm. The
mock driver does not. `farm config doctor` surfaces this as a critical
finding. See [config-reference.md](config-reference.md#arm).

## FARM-E1009

> Config not found at `<path>`. fix: `farm config init`

`farm` looked for `~/.farm/config.yaml` (or `--config`) and found nothing.
Run `farm config init` to scaffold the template, or pass `--config <path>`.

## FARM-E1010

> Required env var `<name>` is not set. fix: `export <name>=...`

A `${...}` interpolation in `config.yaml` references an env var that is not
present in the current shell. The Edge Agent will not silently substitute
an empty string. See [config-reference.md](config-reference.md#env-vars).

## Canonical URLs

Each code has a canonical online page mirrored from this file:

- <https://farm.dev/errors/E1001>
- <https://farm.dev/errors/E1002>
- <https://farm.dev/errors/E1003>
- <https://farm.dev/errors/E1004>
- <https://farm.dev/errors/E1005>
- <https://farm.dev/errors/E1006>
- <https://farm.dev/errors/E1007>
- <https://farm.dev/errors/E1008>
- <https://farm.dev/errors/E1009>
- <https://farm.dev/errors/E1010>
- <https://farm.dev/errors/E2001>
- <https://farm.dev/errors/E3001>
- <https://farm.dev/errors/E3002>
