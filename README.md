# FARM

UF850 sim + teleop harness. CS153 final project.

Laptop side: MuJoCo sim, webviz-style browser dashboard, ROS-TCP-Endpoint
bridge wired up so a future Quest VR client can plug in unmodified. No
planner/policy stack today — the rewrite stripped that to focus on a clean
teleop surface.

## Quickstart

Need Python 3.12.

```bash
git clone <this repo> && cd CS153
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./farm-shared -e ./farm-edge-agent
pip install mujoco aiohttp aiohttp-cors pillow

farm serve                  # opens the dashboard in your browser
```

The dashboard at http://127.0.0.1:8787/ shows three live MuJoCo camera
feeds (exterior, wrist, topdown), joint bars, a TCP/RPY readout, and
cartesian jog buttons. The Quest teleop bridge listens on `:10000` (ROS-TCP
wire format) — speak it from any client and you can drive the arm.

## Layout

- **`farm-edge-agent/`** — the harness: sim, server (HTTP + SSE), ROS-TCP
  bridge, web dashboard, CLI.
- **`farm-shared/`** — shared error catalog.
- **`farm-cloud/`** — placeholder for future GPU-side deployment.

## Commands

```bash
farm serve                          # daemon + dashboard + ROS-TCP bridge
farm config init                    # scaffold ~/.farm/config.yaml
pytest farm-edge-agent/tests
ruff check .
```

## HTTP API

```text
GET  /                       — dashboard
GET  /v1/world               — current sim snapshot
GET  /v1/world/stream        — SSE stream of snapshots
POST /v1/teleop/jog          — {axis, sign, step_mm?, step_rad?}
POST /v1/teleop/home         — drive arm to HOME_JOINTS
POST /v1/teleop/gripper      — {state: "open" | "closed"}
GET  /v1/cameras/{name}.jpg  — exterior | wrist | topdown
```

## ROS-TCP bridge

Listens on TCP `:10000`. Speaks the
`Unity.Robotics.ROSTCPConnector` wire format (4-byte topic length + UTF-8
topic + 4-byte body length + body). Today it accepts `/q2r_*` Quest
publishers and pumps `/joint_states` outbound at 10 Hz. See
`src/farm_edge_agent/ros_bridge/` for the full topic schema list.
