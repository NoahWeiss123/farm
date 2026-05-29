# FARM

**F**ine-tuned **A**rm **R**obot **M**anipulation — a teleoperation +
imitation-learning harness for the UFACTORY UF850. CS153 final project.

Record arm demonstrations in VR, train a π0.5 vision-language-action policy on
them, and run the policy back on the arm:

```
   ┌─────────┐   teleop    ┌──────────────┐  export   ┌──────────┐  fine-tune  ┌──────────┐  serve   ┌──────────┐
   │  Quest  │ ──────────▶ │  farm serve  │ ────────▶ │ LeRobot  │ ──────────▶ │   π0.5   │ ───────▶ │   arm    │
   │  (VR)   │  ROS-TCP    │  (record)    │           │ dataset  │  H100s      │  policy  │  eval    │  (UF850) │
   └─────────┘             └──────────────┘           └──────────┘             └──────────┘          └──────────┘
```

The arm is a 6-DoF UF850 + parallel gripper with two RealSense cameras (a base
view and a wrist view). A MuJoCo sim stands in when no hardware is attached.

## Repository layout

```
farm/
├── teleop/            Data collection — drive the arm + record demos
│   ├── edge-agent/      the daemon (farm serve): sim + xArm backend, HTTP/SSE,
│   │                    ROS-TCP bridge, recorder, episode review, CLI
│   └── quest/           Quest 3 VR client (Unity) — publishes controller poses
├── ui/                Browser dashboard + episode-review app (single-file HTML)
├── model/             The policy: dataset export, training, eval, serving
│   ├── export_lerobot.py, analyze_dataset.py, eval_pi05.py, …
│   ├── cluster/         H100 fine-tuning — three architectures (see below)
│   ├── cloud/           optional Modal-hosted policy server
│   └── rtc/             Real-Time Chunking dev probes (motion smoothness)
├── shared/            Shared error catalog (farm_shared)
└── datasets/          Recordings + LeRobot exports (gitignored — on the HF Hub)
```

## Quickstart — run the daemon

Needs Python 3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./shared            # the shared error catalog
pip install -e ./teleop/edge-agent # the daemon (pulls farm_shared, mujoco, aiohttp…)
pip install mujoco                 # sim backend

farm serve            # daemon + dashboard + ROS-TCP bridge; opens the browser
```

Open **http://127.0.0.1:8787/** — live camera tiles (base + wrist), joint bars,
a TCP/RPY readout, cartesian jog buttons, the recorder, and a policy-eval panel.
**`/review`** is the episode-review + clip app for curating recordings. The
Quest teleop bridge listens on `:10000` (ROS-TCP wire format).

## Training a policy

1. **Record** demos with `farm serve` + the Quest client; curate them in `/review`.
2. **Export** to a LeRobot dataset and push to the Hub:
   ```bash
   python model/export_lerobot.py --src datasets/dataset3 --out datasets/lerobot/farm_uf850_bottle
   python model/analyze_dataset.py        # audit alignment, smoothness, gripper, tasks
   ```
3. **Fine-tune** π0.5 on the H100 cluster. Three interchangeable architectures,
   all comparable on the same data + action contract:

   | Config | Method | GPUs | Idea |
   |---|---|---|---|
   | `pi05_farm_uf850` | full fine-tune | 8 | max capacity; overfits small data |
   | `pi05_farm_uf850_lora` | LoRA | 1 | freezes the base; preserves it, can under-adapt |
   | `pi05_farm_uf850_gse` | **GSE** (VLA-GSE) | 1 | SVD spectral experts — preserve *and* adapt |

   See **[`model/cluster/README.md`](model/cluster/README.md)** for the runbook
   and **[`model/FINDINGS.md`](model/FINDINGS.md)** for why the deployed full-FT
   model underperforms and how the three compare.
4. **Serve + evaluate**: `model/cluster/serve_pi05.sbatch` runs the policy
   server; `python model/eval_pi05.py` reads observations from `farm serve` and
   drives the arm.

## Common commands

```bash
farm serve                       # daemon + dashboard + ROS-TCP bridge
farm config init                 # scaffold ~/.farm/config.yaml
pytest teleop/edge-agent/tests   # daemon tests (deterministic, no GPU)
pytest shared/tests              # shared-catalog tests
ruff check .                     # lint
```

## HTTP API (selected)

```text
GET  /                          dashboard          GET  /review               episode-review app
GET  /v1/world  /v1/world/stream  snapshot + SSE   GET  /v1/cameras/{base,wrist}.jpg
POST /v1/teleop/jog|home|gripper|joints            POST /v1/teleop/estop[/clear]
POST /v1/policy/run|stop · prompt · heartbeat      GET  /v1/episodes …          record + review
```

## ROS-TCP bridge

Listens on TCP `:10000`, speaking the `Unity.Robotics.ROSTCPConnector` wire
format (4-byte topic length + UTF-8 topic + 4-byte body length + body). It
accepts `/q2r_*` Quest publishers and pumps `/joint_states` outbound at 10 Hz.
See `teleop/edge-agent/src/farm_edge_agent/ros_bridge/` for the topic schemas.
