# FARM

Agent harness for Pi0.5 + UFactory 850. CS153 final project.

High-level task → OpenAI decomposes into subtasks → each subtask runs through Pi0.5 → safety gates → arm executes.

Everything runs in sim by default, no hardware required.

## Quickstart

Need Python 3.12 and an OpenAI key.

```bash
git clone <this repo> && cd CS153
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./farm-shared -e ./farm-edge-agent
pip install mujoco openai opencv-python-headless aiohttp aiohttp-cors pillow

cp .env.example .env
$EDITOR .env   # OPENAI_API_KEY=sk-...

export $(grep -E '^[A-Z_]+=' .env | xargs)
farm serve
```

Try via curl:

```bash
curl -X POST http://127.0.0.1:8787/v1/runs -d '{"task": "pick the red block and place it on the cup"}'
```

## Layout

- **`farm-edge-agent/`** — the agent harness. Drivers (MuJoCo sim, xArm real), Pi0.5 policy client, OpenAI task planner, safety enforcer, skill library, HTTP daemon.
- **`farm-shared/`** — shared error catalog.
- **`farm-cloud/modal/`** — Pi0.5 inference server (Modal, ~24 GB GPU).

## How a run flows

POST `/v1/runs` → RunSupervisor picks backend → **GPT path**: OpenAI decomposes task into skill calls, each skill emits action chunks → **Pi0.5 path**: camera + joints + prompt → Modal endpoint → joint delta actions at 20 Hz → SafetyEnforcer gates every chunk (envelope, velocity, singularity) → Driver executes on sim or real arm.

## Commands

```bash
farm serve                          # daemon on :8787
pytest farm-edge-agent/tests        # 106 tests
ruff check .
```

## Adding a skill

Write a function in `farm_edge_agent/skills/library.py`, call `register(SkillSpec(...))`, restart `farm serve`. The planner picks it up automatically.
