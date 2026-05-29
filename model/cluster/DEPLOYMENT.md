# Running the trained π0.5 policy (FARM UF850)

Everything you need to **serve the trained model and drive the arm**, set up to the
point right before you activate a GPU. The only command that touches a GPU is
`sbatch serve_pi05.sbatch` (step C) — stop there until you're ready to run live.

## The model

Training (job 113, full fine-tune) produced the final policy at **`step-19999`**,
public on the Hub: **`NoahWeiss/farm_uf850_pi05`** (revision `step-19999`).
Intermediate checkpoints `step-5000 / 10000 / 15000` are also there — **the final
step isn't always the best policy**, so eval a few and pick the winner (see
"Checkpoint selection").

## Architecture

The robot/sim and the policy live in separate environments and talk over a
WebSocket (openpi's remote-inference design — heavy GPU off-robot, no dependency
clashes with the robot stack):

```
  LAPTOP (or robot host)                         CLUSTER (1× H100)
  ┌─────────────────────────────┐                ┌──────────────────────────┐
  │ farm serve   (FARM daemon)   │   obs (state   │  serve_policy.py :8000   │
  │  • UF850 arm + RealSense cams│   + 224² imgs) │  • loads step-19999      │
  │  • /v1/world, /v1/cameras    │ ─────────────► │  • normalizes state      │
  │  • /v1/teleop/joints (apply) │ ◄───────────── │  • returns 10-action     │
  │ eval_pi05.py (client)        │  action chunk  │    chunk (abs joints)    │
  └─────────────────────────────┘                └──────────────────────────┘
            ▲  localhost:8787                         ▲  ws://…:8000
            └── runs the control loop ────────────────┘ (kubectl port-forward)
```

- **Policy server** (cluster GPU): `serve_policy.py` loads `step-19999`, listens on `:8000`.
- **FARM daemon** (laptop): `farm serve` — owns the arm + cameras, exposes obs and an apply endpoint.
- **Eval client** (laptop): `eval_pi05.py` — the control loop. Pulls obs from the daemon, queries the policy, applies the returned joint targets.

## Setup — GPU-free up to step C

### A. Model — done ✓
Trained and on HF (`step-19999`). The serve job auto-resolves the checkpoint
(local training output → previously-downloaded path → HF download), so nothing to stage.

### B. Stage the launcher files to the pod (laptop)
```bash
USER_NAME=<your-cluster-username>
POD=$(kubectl get pod -n slurm -l stanford/user=${USER_NAME} -o jsonpath='{.items[0].metadata.name}')
kubectl cp tools/cluster/serve_pi05.sbatch slurm/$POD:/home/$USER_NAME/farm-train/serve_pi05.sbatch -c login
```
(`.hf_env` from training is already on the pod and is only used if the HF-download fallback runs.)

### C. ⚡ ACTIVATE THE GPU — launch the policy server (login pod)
**This is the line to stop before until you want to run live.**
```bash
cd ~/farm-train && sbatch serve_pi05.sbatch
squeue -u $USER                       # note the node (e.g. slinky-2)
tail -f ~/farm-train/serve-<jobid>.out   # wait for ">>> serve_policy.py on :8000" + model load
```
Batch-1 inference of π0.5 fits easily on one H100. First load restores ~12.5 GB of
params (~10 s) + a one-time JIT compile of the inference graph (~1–2 min); after that
each `infer()` is ~80–100 ms.

### D. Forward the port to your laptop (two hops — no SSH on this cluster)
```bash
# In the login pod: forward the worker's :8000 to the login pod
JOB=$(squeue -u $USER -h -o '%i' -n serve-pi05 | head -1)
WORKER=$(squeue -j $JOB -h -o '%R')
kubectl port-forward -n slurm pod/$WORKER 8000:8000 &
# On the laptop: forward the login pod's :8000 to localhost
kubectl port-forward -n slurm $POD 8000:8000
```
Now `ws://127.0.0.1:8000` on the laptop reaches the policy server.

### E. Start the FARM daemon (laptop / robot host)
```bash
source .venv/bin/activate && farm serve     # arm + cameras on http://127.0.0.1:8787
```

### F. Run the control loop
```bash
# Dry-run first (prints joint targets, does NOT move the arm) — always verify before live:
python tools/eval_pi05.py \
    --task "Picking up the bottle and placing it on the box" \
    --policy-url ws://127.0.0.1:8000 \
    --daemon-url http://127.0.0.1:8787 \
    --dry-run
# Then drop --dry-run to execute on the arm. --list-tasks prints the trained prompts.
```

## High-quality inference settings (and why)

These defaults in `eval_pi05.py` are research-backed (openpi remote-inference guide +
Physical Intelligence's real-time-chunking work); the rationale matters if you tune them:

| Setting | Value | Why |
|---|---|---|
| **Control rate** | **30 Hz** | Must match the training timestep — each action = state at t+1/30 s. Faster playback makes the PD tracker lag and miss grasps by 1–2 cm. |
| **Action chunking** | open-loop, **10-step chunks**, re-inferred **asynchronously** (pipelined) | π0.5 predicts a 10-action chunk (0.33 s at 30 Hz). The next chunk is generated *while* the current one executes, so the ~80–120 ms inference is fully hidden (chunk ≫ latency). Re-plan cadence (`--steps-per-chunk`) can drop to ~5 for more reactivity; with async it's nearly free. |
| **Image preprocessing** | `resize_with_pad` → **224×224**, uint8 | Matches π0.5's training pipeline exactly. A plain resize *squishes* the 4:3 RealSense frame to 1:1 — the model never saw distorted images and grasp accuracy drops. |
| **State** | passed **unnormalized** | The server normalizes via the checkpoint's `norm_stats`. Don't double-normalize. |
| **Action space** | **absolute** joint targets | We trained π0.5 with absolute actions (`use_delta_joint_actions=False`); the client's default `--action-mode absolute` matches. |
| **Prompt** | exact trained string | π0.5 is language-conditioned — use a string from `--list-tasks` (the two bottle tasks), not a paraphrase. |

### The SOTA upgrade path: Real-Time Chunking (RTC)
If you ever push to **higher-latency or more dynamic** tasks, Physical Intelligence's
**Real-Time Chunking** is the state of the art for flow-matching VLAs like π0.5: it
generates the next chunk asynchronously *and* guides it to align with the already-executed
portion of the previous chunk, giving smooth motion even at >300 ms latency. **We don't
need it here** — our inference (~80–120 ms incl. LAN) is far below the 0.33 s chunk, so the
plain pipelined approach already hides the latency — but it's the path to take if you add
faster/reactive tasks. Refs: [pi.website](https://www.pi.website/research/real_time_chunking),
[paper](https://arxiv.org/pdf/2506.07339), [LeRobot RTC](https://huggingface.co/docs/lerobot/rtc).

## Checkpoint selection
The final checkpoint isn't guaranteed best (overfitting late in a small 2-task run is
possible). Serve each of `step-5000 / 10000 / 15000 / 19999` (swap `--policy.dir` /
the HF revision), run several trials per bottle task, and keep the highest success rate.
To serve an earlier one, set the sbatch's checkpoint or download e.g.
`hf download NoahWeiss/farm_uf850_pi05 --include 'step-15000/*' --local-dir ~/farm-train/checkpoints/pi05_step15000`.

## Safety
- **Always `--dry-run` first** — confirms the joint targets look sane before any motion.
- The FARM dashboard's **Stop** / `/v1/teleop/estop` halts the arm; the client checks the
  e-stop flag each step and aborts.
- Keep the workspace clear and a hand on the e-stop for the first live runs.

## Cost / teardown
The serve job holds **1 GPU** for its walltime (4 h default). **`scancel <jobid>`** when
done so you stop consuming the cluster — the server doesn't exit on its own.
