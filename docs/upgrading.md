# Upgrading

The Edge Agent is pip-installed on your machine; the cloud ships on its
own cadence. Drift is inevitable. This page is the reference for what is
compatible with what, and how to move forward when something is not.

## How versioning works

Two versions matter:

- **Edge Agent package version** — the `farm-edge-agent` package on PyPI.
  Semantic versioning. Printed by `farm version`. Informational; the
  dispatcher does not refuse a connection based on package version.
- **Wire protocol version** — semver, separate from the package version.
  Bumped on breaking changes to the obs/action chunk schemas, control
  messages, or run-record format. Used in the handshake. The dispatcher
  refuses a connection if there is no version overlap and emits
  [FARM-E1006](errors.md#farm-e1006).

A single Edge Agent package may speak multiple wire protocol versions;
`farm version --json` prints the range it supports.

## Protocol version table

| Wire protocol | First Edge Agent | Status | Notes |
|---|---|---|---|
| `1.0` | `farm-edge-agent==0.1.0` | Deprecated | Obs/action schema before run-record signing was added. Removed `2026-03-01`. |
| `1.1` | `farm-edge-agent==0.2.0` | Deprecated | Adds HMAC-signed run records. Sunsets `2026-08-01`. |
| `1.2` | `farm-edge-agent==0.3.0` | **Current — Phase-MVP** | Adds `calibration_hash` field to run record + handshake. Dispatcher minimum. |

Phase-MVP ships with wire protocol `1.2`. The dispatcher accepts `1.2`
and rejects anything lower. New Phase-Product protocol versions land
post-CS153 and will appear in this table when they do.

## Upgrading the Edge Agent

```bash
pip install -U farm-edge-agent
farm version
```

`farm version` after the upgrade should print a wire protocol version at
or above the dispatcher's minimum. The next `farm start` will re-handshake
cleanly.

To make a session self-heal, pass `--auto-update`:

```bash
farm start --auto-update
```

On protocol mismatch, the Edge Agent runs `pip install -U farm-edge-agent`,
re-exec's itself, and reconnects. Off by default — manufacturing users do
not want their controller upgrading itself mid-shift. The public sandbox
sets it on.

## Migration notes

### `1.1` → `1.2`

- **What changed.** Run records now carry a `calibration_hash` field
  alongside the existing `intrinsics_path`. The Edge Agent refuses to
  start a run with stale calibration unless `--accept-calibration` is
  passed ([FARM-E1002](errors.md#farm-e1002)).
- **What to do.** Run `farm calibrate` against your current camera mount
  before your next run. Update any tooling that parses run records to
  read the new field; the old `intrinsics_path` still appears.
- **`farm verify` behaviour.** Records produced under `1.1` verify
  against a `1.2` `farm.lock` as long as the lock omits
  `calibration_hash`. Adding a hash to the lock retroactively will fail
  verification on older records — start a new lock instead of mutating
  the old one.

### `1.0` → `1.1`

- **What changed.** HMAC signing of run records. New `signature` block
  appended to every record.
- **What to do.** Generate a workspace signing key via the dashboard.
  Existing run records are not retroactively signed; treat them as
  pre-signing artifacts.

## Lock files and protocol versions

A workspace can pin protocol + backend digests in `farm.lock`:

```yaml
# farm.lock
edge_agent: 0.3.0
protocol: 1.2
backends:
  - id: pi05-ufactory-ft-v1
    digest: sha256:9a3...
  - id: classical-planner
    version: 1.0
calibration_hash: sha256:a8b...
```

`farm verify <run-id>` checks the record against this lock. Exit codes
are CI-friendly: `0` on match, `1` on signature failure, `2` on lock
drift. The verification is HMAC against the workspace signing key — a
holder of that key can confirm a record was not tampered with. This is
research-grade reproducibility, not third-party-auditable signing; the
latter is Phase-Product (see DESIGN.md → Reproducibility).

## What to do when the dispatcher rejects your connection

Order of operations:

1. Read the error. [FARM-E1006](errors.md#farm-e1006) prints the
   detected agent protocol version and the dispatcher minimum.
2. `pip install -U farm-edge-agent`.
3. Re-run `farm version` and confirm the protocol range now overlaps.
4. Re-run `farm start`.

If the same error reappears after upgrade, your pip cache may have
served an old wheel — clear it with `pip cache purge` and reinstall.

## What to do when `farm verify` fails

The exit code tells you which class of failure:

- **`1` — signature mismatch.** The record's HMAC does not match the
  workspace signing key. Either the record was tampered with after
  emission, the key was rotated since the run, or the lock points at the
  wrong workspace. Inspect the dashboard for the run's workspace and
  signing key version.
- **`2` — lock drift.** The record is authentic but a pinned digest does
  not match. The error includes which `backends[].digest` or
  `calibration_hash` drifted. Re-pin the lock to current digests only if
  the drift was intentional; otherwise treat the result as a real
  reproducibility miss and investigate.

See also: [errors.md](errors.md), [safety.md](safety.md#calibration-drift-detection),
[python-api.md](python-api.md#run-records).
