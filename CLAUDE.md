# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

FARM — a UF850 teleop + π0.5 imitation-learning harness. CS153 final project.

The loop: VR-teleop the arm to record demos (`farm serve` + the Quest
client) → export to a LeRobot dataset → fine-tune π0.5 on the H100
cluster → serve the checkpoint and drive the arm from the policy. The
laptop side is a clean MuJoCo sim (stand-in for the arm), an aiohttp
dashboard + episode-review app, and a ROS-TCP-Endpoint bridge the Quest
client speaks to. The model pipeline (export, cluster training, eval)
lives in `model/` — see `model/README.md` and `model/FINDINGS.md`.

Note: an *older* planner stack (GPT planner → Pi0.5 → safety gates → arm)
was deleted on 2026-05-25; the current π0.5 work is an imitation-learning
fine-tune in `model/`, unrelated to that removed code (see "What was
deleted" below).

## Common commands

```bash
source .venv/bin/activate
# install in two steps — the edge-agent's direct-reference dep on farm-shared
# conflicts with a single combined `pip install -e` invocation
pip install -e ./shared
pip install -e ./teleop/edge-agent

# run the local daemon (sim + dashboard + ROS-TCP bridge)
farm serve

# tests
pytest teleop/edge-agent/tests

# lint
ruff check .
ruff check --fix .
```

## Architecture

Two Python packages, plus the Quest client, the model tooling, and an
optional cloud server.

### Packages & components

- **`teleop/edge-agent/`** — the daemon (`farm serve`). Owns: the MuJoCo sim,
  the real-arm (xArm) backend, the HTTP/SSE server (aiohttp), the
  ROS-TCP-Endpoint wire bridge, the dashboard + episode-review app, the
  teleop recorder, the CLI.
- **`shared/`** — shared error catalog (`ErrorCode` enum with
  format-string templates).
- **`teleop/quest/`** — Quest VR teleop client (Unity); publishes controller
  poses over ROS-TCP. Fix it in place (don't resurrect the old standalone
  collector project).
- **`ui/`** — the browser dashboard + episode-review app (`index.html`,
  `review.html`). Served by the daemon from the repo-level `ui/` folder.
- **`model/`** — model workstream: `export_lerobot.py`, `analyze_dataset.py`,
  `eval_pi05.py`, and `cluster/` (the three π0.5 fine-tune configs — full FT,
  LoRA, GSE — + serve) + `cloud/` (optional Modal server). See `model/README.md`.

### Key directories

```
teleop/edge-agent/src/farm_edge_agent/
├── cli/           # Click CLI: farm {serve, config, version}
├── config/        # YAML config loading + validation
├── drivers/       # base protocol + real-arm xArm driver
├── sim/           # lean MuJoCo backend (Sim, jog, render, IK)
├── ros_bridge/    # ROS-TCP-Endpoint-compatible TCP listener + message codecs
├── server/        # aiohttp daemon (app, supervisor, event bus)
└── errors.py
```

The dashboard is **not** inside the package — it lives at the repo-level
`ui/` folder and the daemon serves it from there (`UI_DIR` in `server/app.py`).

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

Daemon work happens in `teleop/edge-agent/`; model/training work in `model/`
(the cluster scripts patch a separate openpi checkout — they don't import
the daemon). The error catalog in `shared/` rarely changes; if you add
a new error code there, update the severity map in
`farm_edge_agent/errors.py` too. Datasets live on the HF Hub, not in git
(everything under `datasets/` is gitignored).

## What was deleted (2026-05-25 rewrite)

Removed during the laptop-side rewrite: the GPT **planner**, the old Pi0.5
*planner* client, skill library, safety gates, run loop, recovery
primitives, and the original 834-line MuJoCo SimDriver. Don't resurrect
that deleted code in place; design fresh.

(The current π0.5 work in `model/` is unrelated — it's a fresh imitation-
learning fine-tune of the policy on teleop demos, not the removed
planner→policy→safety stack.)
