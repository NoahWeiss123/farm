# Errors

Every error follows the same shape:

```
[FARM-Exxxx] <one-line description> — fix: <command or action>
```

The structured form is reproduced in JSON output (`--json`) and in run records. Codes are stable across the wire protocol. New failure modes get new codes; existing codes are not renumbered.

## Quick reference

| Code | Surface | One-liner |
|---|---|---|
| FARM-E1001 | CLI | Camera device not found |
| FARM-E1002 | CLI | Calibration is stale |
| FARM-E1003 | CLI + UI | GPU container cold-starting |
| FARM-E1004 | CLI | API key rejected |
| FARM-E1005 | CLI + UI | Dispatcher WebSocket dropped |
| FARM-E1006 | CLI | Wire protocol version mismatch |
| FARM-E1007 | CLI | Network probe failed |
| FARM-E1008 | CLI | Driver requires `arm.ip` in config |
| FARM-E1009 | CLI | Config file not found |
| FARM-E1010 | CLI | Required env var missing |
| FARM-E2001 | CLI | Capability card validation failed |
| FARM-E3001 | Edge Agent | Safety envelope violation |
| FARM-E3002 | Edge Agent | Watchdog timeout |

The canonical templates live in `farm_shared.errors.ErrorCode`. Render with `format_error(code, **slots)`, never f-string the enum directly.

## FARM-E1001

> No camera found at `<device>`.

The configured `camera.<view>.device` path does not enumerate. Most often this is a `/dev/video0` vs `/dev/video2` mistake after a USB hub re-enumerated.

## FARM-E1002

> Calibration is `<N>` days old.

The Edge Agent hashes `camera.<view>.intrinsics` at run start. If older than 24 hours, the run is refused unless `--accept-calibration` is passed. Camera-mount bumps are the #1 silent failure mode in lab robotics.

## FARM-E1003

> GPU container cold-starting (typical 8 to 25s). Holding the run open; arm will move when ready.

Informational. The cloud inference container had to spin up.

## FARM-E1004

> API key rejected.

The dispatcher rejected the workspace key during handshake. Either the key was rotated, or `${FARM_API_KEY}` resolves to empty.

## FARM-E1005

> Dispatcher WebSocket dropped after `<N>` s. Auto-reconnecting; arm halted in place.

Connection lost mid-run. Watchdog halted the arm in place; dispatcher persisted state up to the last completed chunk.

## FARM-E1006

> Edge Agent v`<X>` detected, Dispatcher requires v`<Y>`+.

Wire-protocol mismatch at handshake. The dispatcher refuses the connection rather than risk silent action-schema drift.

## FARM-E1007

> Network probe FAILED: WebSocket upgrade blocked.

DNS resolved but the WebSocket upgrade was blocked, typically by a captive portal, corporate proxy, or aggressive firewall. `FARM_RELAY=on` tunnels over HTTPS long-polling.

## FARM-E1008

> Driver `'<driver>'` requires `arm.ip` in config.

The configured driver (e.g. `xarm`) needs a network address. The mock driver does not.

## FARM-E1009

> Config not found at `<path>`.

`farm` looked for `~/.farm/config.yaml` (or `--config`) and found nothing.

## FARM-E1010

> Required env var `<name>` is not set.

A `${...}` interpolation in `config.yaml` references an env var that's not present. The Edge Agent will not silently substitute an empty string.

## FARM-E2001

> `capability_card.<path>`: `'<value>'` not in allowed set. Did you mean `'<suggestion>'`?

A capability card failed schema validation. Enum mismatches include a `Did you mean` suggestion based on edit distance.

## FARM-E3001

> Safety envelope violation: commanded pose outside workspace. Soft-stopped.

A planned chunk would have put the TCP outside `safety.workspace_envelope`. The Edge Agent soft-stopped the arm before the move executed.

## FARM-E3002

> Watchdog timeout (>1s server silence). Arm halted in place.

The dispatcher's WebSocket went silent for longer than `safety.watchdog_timeout_ms`. The cloud cannot stall the arm in a dangerous configuration.
