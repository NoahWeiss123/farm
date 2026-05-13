# FARM

Foundation for Action-Reasoning Models. A hosted agent harness for robotics foundation models.

CS153 final project. Solo build.

## What this is

FARM sits between "user describes a task" and "robot executes it." It exposes a uniform API across multiple VLA backends (π0.5, Gemini Robotics, classical planner), runs an LLM router that picks the right backend per subtask, and streams telemetry to a live dashboard. Inference runs on Cloudflare GPU containers; the control loop runs locally next to the arm via a pip-installable Edge Agent.

Demo target: a UFactory 850 6-DOF arm running a fine-tuned π0.5 on a colored-block stacking task, with a classical-planner fallback for graceful recovery.

Full design: [DESIGN.md](DESIGN.md). Deferred work: [TODOS.md](TODOS.md).

## Quickstart (when implemented)

```
pip install farm-edge-agent
farm quickstart
```

Targets: 3 min sim arm, 30 min real arm.

## Repo layout

```
farm-edge-agent/   # python package: CLI, drivers, safety, recovery, run records
farm-cloud/        # cloudflare side: Worker (planner + dispatcher), Pages (UI)
farm-shared/       # cross-package contracts: schemas, error catalog, protocol versions
docs/              # config/cli/python-api/errors/hardware/safety/faq reference
tasks/             # task specs the orchestrator works through
bin/               # orchestrator scripts
AGENTS.md          # rules for agents working in this repo
```

## Working on this

Read [AGENTS.md](AGENTS.md) before making changes. The commit-message style is enforced by `.githooks/prepare-commit-msg`.
