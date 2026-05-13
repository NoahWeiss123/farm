# Config reference

The Edge Agent reads `~/.farm/config.yaml` on every CLI invocation. Override
the path with `FARM_CONFIG=/path/to/config.yaml` or the global
`--config <path>` flag. `farm config init` scaffolds a fresh file;
`farm config doctor` validates it and prints fix commands; `farm config show`
prints the effective config with secrets redacted.

## Precedence

For any given key, the value used is the first that resolves:

1. CLI flag (e.g. `--workspace my-lab`)
2. Environment variable (e.g. `FARM_API_KEY`)
3. Inline value in `~/.farm/config.yaml`
4. Compiled-in default

Env-var references inside the config file (`${FARM_API_KEY}`) are resolved at
load time. Missing env vars in `${}` references are a fatal config error
([FARM-E1004](errors.md#farm-e1004)).

## Full example

```yaml
# ~/.farm/config.yaml
api_key: ${FARM_API_KEY}
workspace: my-lab
driver: lerobot-mock          # xarm | franka | lerobot-mock

arm:
  ip: 192.168.1.213           # xarm/franka only; required when driver != lerobot-mock

camera:
  wrist:
    device: /dev/video0
    intrinsics: ./calib/wrist.yaml
  overhead:                   # optional
    device: /dev/video1
    intrinsics: ./calib/overhead.yaml

safety:
  workspace_envelope: ./envelope.yaml
  velocity_cap_mps: 0.25
  watchdog_timeout_ms: 1000

telemetry:
  upload_frames: true
```

## Key reference

### `api_key` *(string, required)*

Workspace-scoped API key issued by `farm login` or `farm quickstart`. Inline
or via `${FARM_API_KEY}`. Env-var form is preferred on shared machines.
Rejected keys raise [FARM-E1004](errors.md#farm-e1004).

### `workspace` *(string, optional)*

Workspace name. Defaults to the first workspace bound to the key. In
Phase-MVP this is a single hardcoded constant per key; the field exists so
the Phase-Product split is mechanical.

### `driver` *(enum, default `lerobot-mock`)*

Arm driver. One of `xarm`, `franka`, `lerobot-mock`. See
[hardware.md](hardware.md) for the compatibility matrix. Drivers other than
the three listed are rejected at config-load time.

### `arm.ip` *(string, required for `xarm` and `franka`)*

IPv4 address of the arm controller. Ignored for `lerobot-mock`.

### `camera.wrist.device` *(string, required when a real arm is configured)*

V4L2 device path on Linux (`/dev/video0`), AVFoundation index on macOS
(`0`), or DirectShow device name on Windows. `farm doctor cameras` lists
what the host actually has.

### `camera.wrist.intrinsics` *(path, required when a real arm is configured)*

Path to a YAML calibration file. Written by `farm calibrate`. The Edge Agent
hashes this file at run start and refuses to start if its mtime is older
than 24 hours unless `--accept-calibration` is passed
([FARM-E1002](errors.md#farm-e1002)).

### `camera.overhead.*` *(object, optional)*

Second camera viewpoint. Same fields as `camera.wrist`. Omit the block if
you only have one camera.

### `safety.workspace_envelope` *(path, default = conservative cube)*

Axis-aligned bounding box of allowed TCP poses. If omitted, a 40 cm cube
centered in front of the arm is used. Out-of-envelope poses are soft-stopped
and surfaced as [FARM-E3001](errors.md#farm-e3001). See
[safety.md](safety.md#workspace-envelope).

### `safety.velocity_cap_mps` *(float, default 0.25)*

Hard ceiling on TCP linear velocity, in metres per second. Joint-space
chunks that imply motion above this cap are clamped and logged.

### `safety.watchdog_timeout_ms` *(int, default 1000)*

Maximum time the dispatcher's WebSocket may be silent before the Edge Agent
halts the arm in place ([FARM-E3002](errors.md#farm-e3002)). Lower this if
your network is fast and steady; raise it if you live behind a flaky tunnel.

### `telemetry.upload_frames` *(bool, default `true`)*

When `true`, camera frames are streamed to the dispatcher for live
rendering and storage. Set to `false` in camera-privacy-sensitive
environments; the run record still captures joint state and action chunks.

## Environment variables

| Variable | Purpose |
|---|---|
| `FARM_CONFIG` | Override the config-file path. |
| `FARM_API_KEY` | Workspace API key. Resolves `${FARM_API_KEY}` in the config. |
| `FARM_RELAY` | `on` tunnels the WebSocket over HTTPS long-polling. See [faq.md](faq.md#my-websocket-wont-connect). |
| `FARM_LOG_LEVEL` | One of `debug`, `info`, `warn`, `error`. Defaults to `info`. |
