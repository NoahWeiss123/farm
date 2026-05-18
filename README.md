# FARM

Foundation for Action-Reasoning Models. A robot agent that takes a prompt and runs it on a UFactory 850 arm. CS153 final project.

Everything runs in sim by default, no hardware required.

## Quickstart

Need Python 3.12, [bun](https://bun.sh), and an OpenAI key.

```bash
git clone <this repo> && cd CS153
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./farm-shared -e ./farm-edge-agent
pip install mujoco openai opencv-python-headless aiohttp aiohttp-cors pillow

cp .env.example .env
$EDITOR .env   # OPENAI_API_KEY=sk-...

export $(grep -E '^[A-Z_]+=' .env | xargs)
farm serve --port 8787 &

bun --cwd farm-cloud/ui install
NEXT_PUBLIC_FARM_API=http://127.0.0.1:8787 bun --cwd farm-cloud/ui run dev
```

Open `http://localhost:3000`. Try:

- `pick the red block and place it on the cup`
- `stack the blue block on top of the green block`
- `pick the green block and place it on the cup, then stack the red block on the blue block`

Runs are saved to `~/.farm/runs/<run_id>/record.jsonl`. The Runs page replays them.

## Layout

`farm-edge-agent/` (Python) is where everything happens. MuJoCo sim driver, safety enforcer, skills, GPT planner with a SQLite cache, aiohttp daemon, JSONL run records.

`farm-cloud/ui/` is the Next.js dashboard. urdf-loader pulls the UF850 mesh, joint stream comes over SSE.

`farm-cloud/worker/` is a Cloudflare Worker that mirrors the planner endpoint. Not needed locally.

`farm-shared/` holds the schemas and error catalog both halves import.

## How a run flows

Dashboard POSTs `/v1/runs`. Supervisor queues it. RunLoop calls the planner: cache hit returns instantly, miss calls OpenAI with the scene and skill catalog. Each plan node dispatches to a skill, the skill spits out action chunks, each chunk goes through SafetyEnforcer (envelope, velocity, singularity) before reaching the driver. MuJoCo steps, joint state events fire back through SSE. Every event lands in the JSONL record.

## Commands

```bash
farm serve                              # daemon on :8787
farm doctor                             # sanity checks
pytest farm-edge-agent/tests            # 244 tests
pytest farm-shared/tests                # 21 tests
ruff check .

bun --cwd farm-cloud/ui run dev         # dashboard
bun --cwd farm-cloud/ui run test
bun --cwd farm-cloud/worker run test    # 51 tests
```

## Adding a skill

Write a Python function in `farm_edge_agent/skills/library.py`, call `register(SkillSpec(...))`, restart `farm serve`. The planner picks it up on the next call.

## What's missing

No real arm yet. The xArm driver shim exists but `farm doctor real-arm` is a stub.

No vision. Object positions come from sim ground truth, not cameras.

No LoRA skill compiler. Plan cache is the only skill layer wired up.

No auth, no billing.

`IMPLEMENTATION_PLAN.md` has the full status grid. `DEPLOY.md` covers Cloudflare.
