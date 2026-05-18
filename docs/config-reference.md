# Config reference

The Edge Agent reads `~/.farm/config.yaml`. Override the path with `FARM_CONFIG=/path/to/config.yaml` or the global `--config <path>` flag. The Pydantic schema lives in `farm_edge_agent.config.schema`.

## Precedence

For any given key, the value used is the first that resolves:

1. CLI flag (e.g. `--workspace my-lab`)
2. Environment variable (e.g. `FARM_API_KEY`)
3. Inline value in `~/.farm/config.yaml`
4. Compiled-in default

Env-var references inside the config (`${FARM_API_KEY}`) are resolved at load time. Missing env vars in `${}` references are a fatal config error ([FARM-E1010](errors.md#farm-e1010)).

## Full example

```yaml
# ~/.farm/config.yaml
api_key: ${FARM_API_KEY}
workspace: my-lab
driver: lerobot-mock          # xarm | franka | lerobot-mock

arm:
  ip: 192.168.1.213           # xarm/franka only

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

## Keys

### `api_key` *(string, required)*

Workspace-scoped API key. Inline or via `${FARM_API_KEY}`. Env-var form is preferred on shared machines.

### `workspace` *(string, optional)*

Workspace name. Reserved for the multi-tenant split; the local daemon ignores it.

### `driver` *(enum, default `lerobot-mock`)*

Arm driver. One of `xarm`, `franka`, `lerobot-mock`. Drivers other than these are rejected at config load.

### `arm.ip` *(string, required for `xarm` and `franka`)*

IPv4 address of the arm controller. Ignored for `lerobot-mock`.

### `camera.wrist.device` *(string, required when a real arm is configured)*

V4L2 device path on Linux, AVFoundation index on macOS, DirectShow device name on Windows.

### `camera.wrist.intrinsics` *(path, required when a real arm is configured)*

Path to a YAML calibration file. The Edge Agent hashes this at run start and refuses to start if its mtime is older than 24 hours unless `--accept-calibration` is passed ([FARM-E1002](errors.md#farm-e1002)).

### `camera.overhead.*` *(object, optional)*

Second camera viewpoint. Same fields as `camera.wrist`.

### `safety.workspace_envelope` *(path, default = conservative cube)*

Axis-aligned bounding box of allowed TCP poses. If omitted, a 40 cm cube centered in front of the arm is used. See [safety.md](safety.md#workspace-envelope).

### `safety.velocity_cap_mps` *(float, default 0.25)*

Hard ceiling on TCP linear velocity, m/s.

### `safety.watchdog_timeout_ms` *(int, default 1000)*

Max dispatcher silence before the Edge Agent halts the arm. Do not set below 250.

### `telemetry.upload_frames` *(bool, default `true`)*

When `true`, camera frames stream to the dispatcher for live rendering. Set `false` in privacy-sensitive environments; the run record still captures joint state and action chunks.

## Environment variables

| Variable | Purpose |
|---|---|
| `FARM_CONFIG` | Override the config-file path. |
| `FARM_API_KEY` | Workspace API key. Resolves `${FARM_API_KEY}` in the config. |
| `OPENAI_API_KEY` | Required by the GPT planner. |
| `FARM_RELAY` | `on` tunnels the WebSocket over HTTPS long-polling. |
| `FARM_LOG_LEVEL` | One of `debug`, `info`, `warn`, `error`. Default `info`. |
