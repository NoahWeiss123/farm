# FARM

A UF850 teleoperation + imitation-learning harness. CS153 final project.

FARM records arm demonstrations by VR teleop, trains a π0.5 vision-language-
action policy on them, and runs the trained policy back on the arm:

```
Quest teleop ─▶ farm serve (record) ─▶ LeRobot dataset ─▶ π0.5 fine-tune (H100s) ─▶ serve ─▶ eval on arm
```

The arm is a UFACTORY UF850 (6-DoF + gripper) with two RealSense cameras (a base
view + a wrist view). A MuJoCo sim stands in for the arm when no hardware is
attached.

## Components

- **`farm-edge-agent/`** — the local daemon (`farm serve`). Owns the MuJoCo sim,
  the real-arm (xArm) backend, the browser dashboard + episode-review app
  (HTTP/SSE), the ROS-TCP-Endpoint bridge the Quest client speaks to, the teleop
  recorder, and the CLI. This is what runs on the laptop during data collection
  and policy eval.
- **`farm-quest/`** — the Quest VR teleop client (Unity). Publishes controller
  poses over ROS-TCP to the daemon, which IK's them onto the arm.
- **`tools/`** — the model workstream: export recordings to a LeRobot dataset,
  fine-tune π0.5 on the cluster (three architectures — see below), and the eval
  client that drives the arm from the trained policy. See `tools/README.md`.
- **`farm-cloud/`** — an optional Modal-hosted policy server (alternative to the
  cluster serve job). See `farm-cloud/README.md`.
- **`farm-shared/`** — shared error catalog used by the daemon.

## Quickstart (daemon + dashboard)

Needs Python 3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./farm-shared -e ./farm-edge-agent
pip install mujoco aiohttp aiohttp-cors pillow

farm serve            # daemon + dashboard + ROS-TCP bridge; opens the browser
```

The dashboard at http://127.0.0.1:8787/ shows the live camera tiles (base +
wrist, physical hardware only — black tiles in sim), joint bars, a TCP/RPY
readout, cartesian jog buttons, the recorder, and a policy-eval panel. The Quest
teleop bridge listens on `:10000` (ROS-TCP wire format). `/review` is the
episode-review + clip app for curating recordings before export.

## Training a policy

1. Record demos with `farm serve` + the Quest client; curate in `/review`.
2. Export to a LeRobot dataset and push to the Hub:
   `python tools/export_lerobot.py --src Dataset3 --out datasets_lerobot/farm_uf850_bottle`
   (audit it with `python tools/analyze_dataset.py`).
3. Fine-tune π0.5 on the H100 cluster — **three interchangeable architectures**,
   all comparable on the same data + action contract:
   - **full fine-tune** (`pi05_farm_uf850`) — max capacity, overfits small data
   - **LoRA** (`pi05_farm_uf850_lora`) — preserves the base, 1 GPU
   - **GSE** (`pi05_farm_uf850_gse`) — VLA-GSE SVD spectral experts, the
     principled middle ground
   See `tools/cluster/README.md` for the runbook and `tools/FINDINGS.md` for why.
4. Serve the checkpoint (`tools/cluster/serve_pi05.sbatch`) and drive the arm
   with `tools/eval_pi05.py`.

## Common commands

```bash
farm serve                          # daemon + dashboard + ROS-TCP bridge
farm config init                    # scaffold ~/.farm/config.yaml
pytest farm-edge-agent/tests        # tests (deterministic, no GPU)
ruff check .                        # lint
```

## HTTP API (selected)

```text
GET  /                       — dashboard            GET  /review — episode review app
GET  /v1/world  /v1/world/stream — snapshot + SSE   GET  /v1/cameras/{base,wrist}.jpg
POST /v1/teleop/jog | home | gripper | joints       POST /v1/teleop/estop[/clear]
POST /v1/policy/{run,stop} | prompt | heartbeat     GET  /v1/episodes ... (record + review)
```

## ROS-TCP bridge

Listens on TCP `:10000`, speaking the `Unity.Robotics.ROSTCPConnector` wire
format (4-byte topic length + UTF-8 topic + 4-byte body length + body). It
accepts `/q2r_*` Quest publishers and pumps `/joint_states` outbound at 10 Hz.
See `farm-edge-agent/src/farm_edge_agent/ros_bridge/` for the topic schemas.
