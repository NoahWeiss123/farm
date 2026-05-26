# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

FARM — a UF850 sim + teleop harness. CS153 final project.

The previous incarnation (GPT planner → Pi0.5 → safety gates → arm) was
deleted on 2026-05-25 in favor of a leaner laptop side: a clean MuJoCo
sim, a webviz-style browser dashboard, and a ROS-TCP-Endpoint bridge
wired up so a Quest VR teleop client can plug in unmodified. The Quest
client itself is a separate workstream.

## Common commands

```bash
source .venv/bin/activate
pip install -e ./farm-shared -e ./farm-edge-agent

# run the local daemon (sim + dashboard + ROS-TCP bridge)
farm serve

# tests
pytest farm-edge-agent/tests

# lint
ruff check .
ruff check --fix .
```

## Architecture

Two Python packages.

### Packages

- **`farm-edge-agent/`** — the daemon. Owns: the MuJoCo sim, the HTTP/SSE
  server (aiohttp), the ROS-TCP-Endpoint wire bridge, the browser
  dashboard, the CLI.
- **`farm-shared/`** — shared error catalog (`ErrorCode` enum with
  format-string templates).

### Key directories

```
farm-edge-agent/src/farm_edge_agent/
├── cli/           # Click CLI: farm {serve, config, version}
├── config/        # YAML config loading + validation
├── drivers/       # base protocol + real-arm xArm driver
├── sim/           # lean MuJoCo backend (Sim, jog, render, IK)
├── ros_bridge/    # ROS-TCP-Endpoint-compatible TCP listener + message codecs
├── server/        # aiohttp daemon (app, supervisor, event bus)
├── web/           # webviz-style dashboard (index.html, single-file)
└── errors.py
```

### How a control loop flows

1. `farm serve` boots the aiohttp daemon, spawns the MuJoCo sim, and
   starts the ROS-TCP bridge listener (default `:10000`).
2. **Dashboard path**: browser fetches `/`, polls `/v1/cameras/*.jpg` at
   ~10 Hz for the camera grid, subscribes to `/v1/world/stream` (SSE) for
   joint bars + TCP pose, POSTs `/v1/teleop/{jog,home,gripper}` for control.
3. **ROS-TCP path**: a Quest client connects to `:10000`, publishes
   `/q2r_right_hand_pose` etc. The bridge decodes via wire-format schemas
   in `ros_bridge/messages.py` and routes to the sim. The bridge publishes
   `/joint_states` outbound at 10 Hz to every connected client.

## Key invariants

- **The Driver protocol** in `drivers/base.py` stays small. The sim
  (`farm_edge_agent.sim.Sim`) implements it; a real-arm driver
  (`farm_edge_agent.drivers.xarm.XArmDriver`) is plug-compatible.
- **Two `FarmError` classes coexist on purpose**: `farm_edge_agent.errors`
  for structured runtime errors, `farm_shared.errors` for the static catalog.
- **`ErrorCode` in `farm_shared.errors`** uses an Enum with `_Spec`
  dataclasses. Use `format_error(code, **slots)` to render.
- **Tests are deterministic.** No `sleep()`, no unseeded randomness in test
  bodies. The sim is constructed with `realtime=False` for test fixtures.
- **ROS-TCP wire format lives in one place** — `ros_bridge/wire.py`. Field
  ordering in `ros_bridge/messages.py` must match the Unity-side
  serializer byte-for-byte, or deserialization silently reads garbage.

## Commit conventions (enforced by hook)

The repo has `core.hooksPath = .githooks` and a `prepare-commit-msg` hook
that strips AI tells.

- Lowercase title, present-tense imperative, under 50 chars.
- **No prefixes.** No `feat:`, `fix:`, etc.
- No `Co-Authored-By: Claude`, no `Generated with [Claude Code]`.
- Good: `add jog endpoint` / `wire ros-tcp bridge` / `fix camera poll loop`

## Scope discipline

Most work happens in `farm-edge-agent/`. The error catalog in
`farm-shared/` rarely changes. If you add a new error code there, update
the severity map in `farm_edge_agent/errors.py` too.

## What was deleted (2026-05-25 rewrite)

Removed during the laptop-side rewrite: the GPT planner, Pi0.5 client,
skill library, safety gates, run loop, recovery primitives, and the
original 834-line MuJoCo SimDriver. If you reach for any of those, you're
probably looking at the wrong shape — they'll come back as separate
modules (or not) once the new sim and teleop bridge settle. Don't
resurrect deleted code in place; design fresh.
