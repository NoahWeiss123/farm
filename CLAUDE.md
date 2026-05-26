# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

FARM — a focused agent harness for driving a UFactory 850 6-DOF arm via Pi0.5 (Physical Intelligence's VLA model). CS153 final project.

The core idea: user gives a high-level task → **OpenAI decomposes it into subtasks** → each subtask runs through **Pi0.5** as a focused prompt → Pi0.5 generates action sequences → actions pass through **safety gates** → UFactory 850 executes.

## Common commands

```bash
source .venv/bin/activate
pip install -e ./farm-shared -e ./farm-edge-agent

# tests (106 passing)
pytest farm-edge-agent/tests

# lint
ruff check .
ruff check --fix .

# run the local daemon (sim + HTTP API on :8787)
farm serve
```

## Architecture

Two Python packages + a Modal deployment for Pi0.5 GPU inference.

### Packages

- **`farm-edge-agent/`** — the agent harness (Python). Owns: the run loop, the OpenAI task planner, the Pi0.5 policy client, the xArm driver, the MuJoCo sim driver, the safety enforcer, recovery primitives, the skill library, the HTTP/SSE server, and the CLI.

- **`farm-shared/`** — shared error catalog (`ErrorCode` enum with format-string templates).

- **`farm-cloud/modal/`** — Pi0.5 inference server deployed on Modal (requires ~24 GB GPU).

### How a run flows

1. User issues `farm run "<prompt>"` or POSTs to `/v1/runs`.
2. RunSupervisor picks a backend: Pi0.5 if `FARM_PI05_ENDPOINT` is set, otherwise GPT planner + skill library.
3. **GPT path**: GptPlanner calls OpenAI to decompose the task into skill calls (pick_and_place, stack, go_to, home). SkillExecutor generates action chunks per skill.
4. **Pi0.5 path**: Pi05Policy sends (camera images + joint state + prompt) to Modal, gets back joint delta actions at 20 Hz.
5. Safety enforcer validates each chunk (envelope, velocity, singularity, e-stop, watchdog).
6. Driver executes on the sim or real arm.

### Key directories

```
farm-edge-agent/src/farm_edge_agent/
├── cli/           # Click CLI: farm {run, start, serve, config, version}
├── config/        # YAML config loading + validation
├── drivers/       # base protocol, SimDriver (MuJoCo), XarmDriver (UFactory 850)
├── policies/      # Pi05Policy — Modal HTTP inference client
├── safety/        # 5 gates: envelope, velocity, singularity, e-stop, watchdog
├── recovery/      # home, open_gripper, relocalize, retry_grasp, abort_safely
├── skills/        # GptPlanner (OpenAI decomposer), SkillExecutor, skill library
├── server/        # aiohttp daemon (HTTP + SSE), RunSupervisor, EventBus
├── run_loop.py    # Orchestrates: plan → execute → safety-gate → drive
└── errors.py      # Structured FarmError (wraps farm_shared.ErrorCode)
```

## Key invariants

- **Safety is non-optional and locally enforced.** Five gates, deterministic, constant-time. No command reaches the arm without passing every gate.
- **Two `FarmError` classes coexist on purpose**: `farm_edge_agent.errors.FarmError` (structured, with severity/exit_code) and the error catalog in `farm_shared.errors`.
- **`ErrorCode` in `farm_shared.errors`** uses an Enum with `_Spec` dataclasses. Use `format_error(code, **slots)` to render.
- **Tests are deterministic.** No `sleep()`, no unseeded randomness.

## Commit conventions (enforced by hook)

The repo has `core.hooksPath = .githooks` and a `prepare-commit-msg` hook that strips AI tells.

- Lowercase title, present-tense imperative, under 50 chars.
- **No prefixes.** No `feat:`, `fix:`, etc.
- No `Co-Authored-By: Claude`, no `Generated with [Claude Code]`.
- Good: `add cli scaffold` / `wire xarm driver` / `fix gripper state on home`

## Scope discipline

Most work happens in `farm-edge-agent/`. The error catalog in `farm-shared/` rarely changes. If you add a new error code there, update the severity map in `farm_edge_agent/errors.py` too.

## Known stubs

These CLI commands print `[FARM] not implemented yet` on purpose:
- `farm run`, `farm start`
