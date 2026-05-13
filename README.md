# FARM

Foundation for Action-Reasoning Models. A self-improving robotics agent
that decomposes natural-language goals, reasons about how to interact with
its environment using whatever tools are available, and turns each
successful run into a reusable skill.

CS153 final project. Solo build.

## What this is

You give the agent a goal in natural language. GPT-5 decomposes it into
sub-tasks. For each sub-task, the agent first checks its **Skill Library**
for a matching skill. If found, it runs the skill (cheap). If not, GPT-5V
describes the scene, the **Affordance Reasoner** lists feasible primitive
actions given the arm and gripper at hand, and a **Code-as-Policy**
executor generates Python that calls those primitives. The code runs under
a deterministic **Safety Enforcer**. On success, the **Skill Compiler**
distills the run into a reusable skill at three layers:

1. Plan cache (cheapest, exact replay)
2. Parameterized code (runs against new initial states)
3. LoRA-fine-tuned π0.5 policy (after enough demos accumulate)

The control loop runs locally on the Edge Agent next to a UFactory 850
6-DOF arm. Routing, skill storage, and the dashboard run on Cloudflare
(Workers + R2 + D1 + Pages). Fine-tuning runs on Modal.

Hardware constraint: the system is built assuming **no specialized
grippers**. Whatever the agent has, it has to figure out how to use.

Full design: [DESIGN.md](DESIGN.md). Build plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Quickstart

```bash
pip install -e ./farm-edge-agent
farm quickstart
```

Targets: 3 min sim arm, 30 min real arm.

## Repo layout

```
farm-edge-agent/   # python package: CLI, run loop, drivers, safety, recovery, skills
farm-cloud/        # cloudflare side: Worker (planner + dispatcher + skill library), Pages (UI)
farm-shared/       # cross-package contracts: schemas, errors, protocol versions
docs/              # user-facing reference (config, CLI, errors, hardware, safety, FAQ)
```

## Working on this

Tests: `pytest farm-shared/tests farm-edge-agent/tests` and `bun --cwd farm-cloud/worker test`.
Dev: `bun --cwd farm-cloud/worker run dev` (worker on :8787) and `bun --cwd farm-cloud/ui run dev` (UI on :3000).
