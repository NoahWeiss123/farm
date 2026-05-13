# Hardware

FARM ships with one tested-and-supported arm, one tested sim driver, and a
small set of contract-stub drivers reserved for Phase-Product. The
compatibility matrix below is authoritative. A machine-readable mirror is
served at `https://farm.dev/hardware.json` (Phase-Product); in Phase-MVP the
markdown here is the source of truth.

## Compatibility matrix

| Arm | Driver | Phase-MVP status | Min firmware | Known issues |
|---|---|---|---|---|
| UFactory 850 | `xarm` | Tested + shipped | v2.1+ | — |
| UFactory xArm 6/7 | `xarm` | Contract-stub | v2.1+ | EE Cartesian frame differs from 850 |
| Franka Panda / FR3 | `franka` | Contract-stub (Phase-Product) | FCI 5.5+ | Gripper kinematics distinct |
| SO-ARM100 / 101 | `lerobot-so` | Contract-stub (Phase-Product) | LeRobot 0.5+ | Reduced workspace envelope |
| Sim (no hardware) | `lerobot-mock` | Tested + shipped | — | — |

"Tested + shipped" means CI runs the Edge Agent against the arm (or its
simulator) on every push to main, with both the happy path and the safety
gates exercised. "Contract-stub" means the driver implements the wire
protocol so the rest of the harness can be exercised, but the
arm-specific kinematics and safety integration are not Phase-MVP work; do
not run them against live hardware.

## Choosing a driver

Set `driver:` in [config.yaml](config-reference.md#driver-enum-default-lerobot-mock).

- **`lerobot-mock`** — no hardware. The sim arm runs in-process and is
  the default after `farm quickstart`. Use this for development, CI, and
  the [3-minute getting-started](getting-started.md) flow.
- **`xarm`** — UFactory arms over the xArm Python SDK and TCP. The tested
  configuration is the 850; the xArm 6/7 contract-stub is included for
  forward compatibility but should not be relied on in Phase-MVP.
- **`franka`** — contract-stub only. Real Franka FCI integration is
  Phase-Product.

## UFactory 850 setup

The tested configuration. Steps:

1. Power the arm and the control box. Wait for the boot LED to settle.
2. Ensure the controller is on the same subnet as your workstation. Default
   IP is `192.168.1.213`.
3. Confirm the hardware e-stop circuit is connected. The Edge Agent refuses
   to begin a run if the e-stop loop is not detected
   ([safety.md](safety.md#physical-e-stop)).
4. Install the wrist camera (Intel RealSense D435 or equivalent) on the
   tool flange. Lock the mount before calibrating — a moved camera between
   calibration and run is invisible to the model.
5. Run the interactive walkthrough:

   ```bash
   farm doctor real-arm
   ```

   It detects the arm, verifies firmware, confirms the e-stop, prompts for
   the camera mount, and offers to run calibration if none is present.

6. Calibrate:

   ```bash
   farm calibrate --camera wrist
   ```

   The Edge Agent hashes the resulting intrinsics file at every run start
   and refuses stale calibrations
   ([FARM-E1002](errors.md#farm-e1002)).

## Cameras

- **Wrist camera** is required for the real-arm path. Intel RealSense D435,
  D435i, or D405 are the tested SKUs. Any V4L2-compatible USB3 RGB camera
  capable of 224×224 @ 10Hz works.
- **Overhead camera** is optional. Same SKU constraints. Configure under
  `camera.overhead` in the config file; omit the block if you only have
  one camera.

Phase-MVP captures RGB only. Depth, force/torque, and tactile sensors are
out of scope (see DESIGN.md → Not in Scope).

## Networking

The Edge Agent talks to the dispatcher over a single WebSocket. The full
preflight is `farm doctor network`. Headline numbers:

- Bandwidth: ≥ 25 Mbps up and down for live frame streaming.
- RTT: < 100 ms to the nearest Cloudflare PoP for chunk-rate execution.
- MTU: ≥ 1280; lower MTU paths often indicate a problematic VPN.

Captive portals, aggressive corporate proxies, and WebSocket-stripping
middleboxes are the common offenders. `FARM_RELAY=on` tunnels over HTTPS
long-polling for those environments; see
[faq.md](faq.md#my-websocket-wont-connect).

## Out-of-scope hardware

For traceability, the following are explicitly not Phase-MVP:

- Bimanual setups (single arm only)
- Mobile bases, gantries
- Force/torque sensors, tactile skins
- Depth-camera-only perception (RGB-D capture is allowed but the
  perception stack ignores the D channel in Phase-MVP)
- Industrial PLC integration

Any of those reaching Phase-Product will appear in a future revision of this
matrix.
