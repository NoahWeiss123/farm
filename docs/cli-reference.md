# CLI reference

The `farm` CLI is a thin wrapper over the Python API
([python-api.md](python-api.md)). Subcommands map 1:1 to module functions on
`farm.Client`. Anything you can do from the terminal you can do from a
notebook.

## Subcommands at a glance

| Command | Purpose |
|---|---|
| [`farm quickstart`](#farm-quickstart) | One-shot setup: sign in, write config, run a canned sim task. |
| [`farm login`](#farm-login) | Browser-based sign-in; stores the API key under `~/.farm/`. |
| [`farm config init`](#farm-config-init) | Scaffold `~/.farm/config.yaml` from template. |
| [`farm config doctor`](#farm-config-doctor) | Validate config; surface fixable errors. |
| [`farm config show`](#farm-config-show) | Print effective config with secrets redacted. |
| [`farm config set`](#farm-config-set) | Mutate a single config value from the CLI. |
| [`farm start`](#farm-start) | Long-running Edge Agent connection to the dispatcher. |
| [`farm run`](#farm-run) | One-shot dispatch from CLI; streams events to stdout. |
| [`farm export`](#farm-export) | Download a run record as JSONL + LeRobot shards. |
| [`farm calibrate`](#farm-calibrate) | Interactive camera intrinsics + hand-eye calibration. |
| [`farm card validate`](#farm-card-validate) | Validate a [capability card](capability-cards.md) against the schema. |
| [`farm doctor`](#farm-doctor) | Run the full preflight (arm, camera, network, protocol). |
| [`farm doctor cameras`](#farm-doctor-cameras) | List cameras, intrinsics, last calibration timestamp. |
| [`farm doctor network`](#farm-doctor-network) | Probe DNS, WebSocket upgrade, RTT, TLS, MTU, throughput. |
| [`farm doctor real-arm`](#farm-doctor-real-arm) | Interactive real-arm setup walkthrough. |
| [`farm verify`](#farm-verify) | Verify a run record against `farm.lock`. |
| [`farm version`](#farm-version) | Print agent version and supported wire protocol versions. |

## Global flags

These work on every subcommand:

| Flag | Effect |
|---|---|
| `--config <path>` | Override `~/.farm/config.yaml`. Same as `FARM_CONFIG`. |
| `--workspace <name>` | Run against a specific workspace. |
| `--json` | Emit machine-readable JSON on stdout instead of human text. |
| `--quiet` | Suppress progress lines; errors still go to stderr. |
| `--auto-update` | If the dispatcher rejects the wire protocol version, `pip install -U` and reconnect. Off by default. See [upgrading.md](upgrading.md). |
| `--accept-calibration` | Suppress the 24-hour calibration-age check. Use with care. |

## farm quickstart

```bash
farm quickstart
```

The TTHW path. Opens a browser to bind a sandbox key, writes
`~/.farm/config.yaml`, connects to the dispatcher, and runs the canned
`wave_hello` task against `lerobot-mock`. See
[getting-started.md](getting-started.md) for a full walkthrough.

## farm login

```bash
farm login
```

Opens a browser, completes device-code sign-in, stores the API key under
`~/.farm/credentials`. Idempotent; re-running rotates the on-disk key.

## farm config init

```bash
farm config init
farm config init --force   # overwrite existing config
```

Scaffolds `~/.farm/config.yaml` with conservative defaults and the
`lerobot-mock` driver. Will not overwrite an existing file unless `--force`
is passed.

## farm config doctor

```bash
farm config doctor
```

Validates the config: schema, required fields per driver, path existence
(intrinsics, envelope), env-var resolution. Prints `OK`, `WARN`, or `ERROR`
with a fix command per finding. Exit code matches the worst severity.

## farm config show

```bash
farm config show
farm config show --json
```

Prints the effective config (file + env vars + CLI overrides). Secrets
(`api_key`) are redacted to `***`.

## farm config set

```bash
farm config set safety.velocity_cap_mps 0.15
farm config set camera.wrist.device /dev/video2
```

Mutates one key in `~/.farm/config.yaml`. Dotted path. Type-checked against
the schema before writing.

## farm start

```bash
farm start
farm start --workspace my-lab
```

Connects the Edge Agent to the dispatcher and stays connected, accepting
runs dispatched from the dashboard. Blocks until Ctrl-C or watchdog
disconnect ([FARM-E1005](errors.md#farm-e1005)).

## farm run

```bash
farm run "pick the red block and stack it on the blue one"
farm run "wave hello" --backend classical
farm run "stack the blocks" --backend auto --task-id exp_42_seed_7
farm run "wave hello" --offline
farm run --resume r_8x2k
```

One-shot dispatch. Streams events to stdout (or JSON with `--json`).

Flags:

- `--backend <id>`: `auto` (default), `pi05-ft`, `gemini-robotics`, `classical`.
- `--task-id <id>`: User-defined identifier stored in the run record.
- `--offline`: Route everything through the local classical-planner backend.
  No cloud calls. See [safety.md](safety.md#offline-mode).
- `--resume <run-id>`: Reconnect to an interrupted run.

## farm export

```bash
farm export r_8x2k
farm export r_8x2k --format lerobot --out ./r_8x2k/
```

Downloads the run record. Default format is JSONL; `--format lerobot` writes
LeRobot-shaped frames. See [python-api.md](python-api.md#exporters).

## farm calibrate

```bash
farm calibrate
farm calibrate --camera wrist
```

Interactive intrinsics + hand-eye calibration. Writes the YAML file
referenced by `camera.<view>.intrinsics`. Updates the calibration hash that
appears in run records.

## farm card validate

```bash
farm card validate ./mycard.yaml
```

Validates a [capability card](capability-cards.md) against
`capability_card.v1.json`. Emits structured errors with allowed-value
suggestions ([FARM-E2001](errors.md#farm-e2001)).

## farm doctor

```bash
farm doctor
```

Runs the full preflight: arm reachable, cameras enumerable, calibration
present and fresh, network probe, protocol negotiation. One verdict at the
end. Equivalent to `farm doctor cameras && farm doctor network && farm doctor real-arm`
in non-interactive mode.

## farm doctor cameras

```bash
farm doctor cameras
```

Lists every camera the host exposes, the intrinsics file (if any), and the
last calibration timestamp. Surfaces [FARM-E1001](errors.md#farm-e1001) when
the configured device is missing.

## farm doctor network

```bash
farm doctor network
```

30-second probe: DNS → WebSocket upgrade → RTT histogram → TLS chain → MTU
→ throughput. Verdict is `OK`, `DEGRADED`, or `FAILED` with a per-failure
fix command ([FARM-E1007](errors.md#farm-e1007)). See
[faq.md](faq.md#my-websocket-wont-connect).

## farm doctor real-arm

```bash
farm doctor real-arm
```

Interactive setup walkthrough against the [hardware matrix](hardware.md):
detect the arm, verify firmware, prompt for IP, confirm the e-stop circuit,
confirm camera mount, run calibration if needed.

## farm verify

```bash
farm verify r_8x2k
farm verify r_8x2k --lock ./farm.lock
```

Verifies a run record's HMAC signature and confirms it was executed against
the locks declared in `farm.lock`. Exit codes are CI-friendly: `0` on
match, `1` on signature failure, `2` on lock drift. See
[upgrading.md](upgrading.md#lock-files-and-protocol-versions).

## farm version

```bash
farm version
farm version --json
```

Prints the Edge Agent package version and the wire protocol versions it
supports. The dispatcher's required protocol version is part of the
handshake ([upgrading.md](upgrading.md)).
