# Full fine-tuning π0.5 on the CS153 H100 cluster

End-to-end workflow for the FARM UF850 dataset. Assumes:

* Dataset on the Hub at **`NoahWeiss/farm_uf850_bottle`** (200 episodes, 2 tasks,
  59,183 frames — see `tools/HUGGINGFACE_UPLOAD.md` to push it)
* Config registered by the patch scripts in **`tools/cluster/`** (step 3
  below applies them) — the live training config is **`pi05_farm_uf850`**,
  a **full fine-tune** of π0.5
* You have cluster access via Omniva CLI (see your CS153 GPU doc)

## 0. Fill in your cluster username (one-time)

Open `.claude/CLAUDE.md` and replace every literal `MY_USERNAME` with your
actual cluster username (the one in your kubeconfig — `kubectl auth whoami`
on your laptop). This is what Claude Code reads when you work on cluster code,
so leaving it as a placeholder is a footgun.

## 1. Shell into the login pod (from your laptop)

```bash
USER_NAME=<your-cluster-username>   # NOT your HF username — your Omniva one
POD=$(kubectl get pod -n slurm -l stanford/user=${USER_NAME} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it -n slurm $POD -c login -- runuser -l ${USER_NAME}
```

## 2. Copy the launcher files to the pod

From your **laptop**, in another terminal. Stage the setup script, the two
config patches, the sbatch, and the checkpoint pusher — the sbatch invokes
`push_checkpoints.py` directly, so it must be on the pod too:

```bash
USER_NAME=<your-cluster-username>
POD=$(kubectl get pod -n slurm -l stanford/user=${USER_NAME} -o jsonpath='{.items[0].metadata.name}')
DST=slurm/$POD:/home/$USER_NAME/farm-train

# Stage everything under ~/farm-train on the pod
kubectl exec -n slurm $POD -c login -- runuser -l ${USER_NAME} -c "mkdir -p ~/farm-train"
kubectl cp tools/cluster/setup.sh                   $DST/setup.sh                   -c login
kubectl cp tools/cluster/patch_openpi_config.py     $DST/patch_openpi_config.py     -c login
kubectl cp tools/cluster/patch_openpi_config_pi05.py $DST/patch_openpi_config_pi05.py -c login
kubectl cp tools/cluster/train_pi05.sbatch          $DST/train_pi05.sbatch          -c login
kubectl cp tools/cluster/push_checkpoints.py        $DST/push_checkpoints.py        -c login
```

## 3. Run setup (on the login pod, inside the runuser shell)

```bash
# In the pod shell from step 1:
cd ~/farm-train
bash setup.sh <PASTE_YOUR_HF_TOKEN_HERE>
```

This:
* Clones `openpi` into `~/farm-train/openpi/`
* Registers the FARM configs by patching openpi's `src/openpi/training/config.py`
  (openpi reads its registry from the `_CONFIGS` list in that file — there's no
  config package to drop a file into). `patch_openpi_config.py` adds the shared
  `LeRobotFarmDataConfig`; `patch_openpi_config_pi05.py` adds the
  `pi05_farm_uf850` full-fine-tune `TrainConfig`. Both are idempotent.
* Verifies the patched `config.py` compiles and the config name landed —
  a cheap login-pod gate that fails *before* you burn a GPU slot.
* Writes a chmod-600 `.hf_env` file with your token so the sbatch can authenticate

## 4. Kick off training

Still on the login pod:

```bash
cd ~/farm-train
sbatch train_pi05.sbatch
# → "Submitted batch job 12345"

squeue -u $USER                  # see queue state
tail -f train-12345.out          # follow output (job id from sbatch)
```

The job:
1. Downloads the dataset from the Hub to `~/.cache/huggingface/lerobot/` on
   first access (one-time, a few hundred MB of video + parquet)
2. Spawns the `nvcr.io#nvidia/pytorch:24.12-py3` container with **4 H100s** +
   32 CPUs (16+ CPUs are for the fast first-time enroot squashfs build, the
   rest feed the data-loader — see your CLAUDE.md)
3. Inside the container: `uv sync` openpi deps (one-time, ~5 min; cached for re-runs)
4. `compute_norm_stats.py` (idempotent, ~1 min once) — this is the only
   data-side step left on the GPU node; everything else (LeRobot formatting,
   video encoding) is already done on your laptop
5. `python scripts/train.py pi05_farm_uf850` — **full fine-tune**, global
   batch 32, 20k steps (~2-2.5 hrs)

**Why 4 GPUs:** a full fine-tune of π0.5 (~3.3B params) does not fit on a
single 80GB H100 — Adam optimizer state alone is ~40GB — so `fsdp_devices=2`
shards the model across 2 GPUs (a memory requirement, not just a speedup). The
other 2 GPUs form a 2nd data-parallel replica: the *same* global batch of 32
trains at ~2× throughput, mathematically identical to the 2-GPU run (same LR,
same final weights), just faster. That's why this is `--gres=gpu:4` rather than
the usual `--gres=gpu:1` default.

Cluster cost: roughly **8-10 H100-hours** (4 GPUs × ~2-2.5h). Walltime is set to
12h in the sbatch so the job won't time out even if the first run's container
build, dataset download, and uv sync all take their slow path.

Checkpoints land under `~/farm-train/openpi/checkpoints/pi05_farm_uf850/farm_uf850_pi05_<jobid>/`.

## 5. Monitor

```bash
# From the pod:
squeue -u $USER                            # is it running?
tail -f ~/farm-train/train-<jobid>.out     # logs
tail -f ~/farm-train/push-<jobid>.out      # checkpoint-pusher logs
sacct -u $USER -S today                    # job history
sshare -u $USER                            # remaining hour budget

# Inside an active job, on a worker (rarely needed):
ssh slinky-X nvidia-smi   # NOT available — use `scontrol show job <id>` instead
```

## 6. After training

The pusher streams each retained checkpoint to
`NoahWeiss/farm_uf850_pi05` during the run, tagged `step-<N>`
(step-5000 / 10000 / 15000 / **step-19999** — openpi names the final
checkpoint at step N-1). To pull the final model anywhere:

```bash
hf download NoahWeiss/farm_uf850_pi05 --include 'step-19999/*' \
    --local-dir ~/farm_pi05_step19999
```

To **serve it and run on the arm**, see `tools/cluster/DEPLOYMENT.md` (full
inference setup + tuned settings). Or copy a checkpoint back to the laptop:
```bash
# From laptop:
kubectl cp slurm/$POD:/home/$USER_NAME/farm-train/openpi/checkpoints/pi05_farm_uf850/farm_uf850_pi05_<jobid>/19999 \
    ~/farm_pi05_v1 -c login
```

Then `tools/eval_pi05.py` reads observations from `farm-edge-agent`'s
`/v1/world` SSE, runs the policy, and drives the arm. See its `--help`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pyxis: importing` hangs >5 min on first run | First-time squashfs build of the 20GB PyTorch image — wait ~3 min with `--cpus-per-task=16`+ |
| `JSON parse error` in container init | You used `docker.io#library/<name>`; switch to `nvcr.io#…` or the bare name |
| `CUDA out of memory` | Drop `batch_size` 32 → 16 in `patch_openpi_config_pi05.py` (re-run setup.sh to re-patch); if still OOM, halve `peak_lr` too |
| `expected 4 GPUs, got N` | The job's JAX device check failed — this config needs `--gres=gpu:4` (2 for FSDP memory + 2 for data parallelism); don't lower it. If you only have 2 GPUs free, set `--gres=gpu:2` *and* drop the assertion to `== 2` — it'll still run, just ~2× slower |
| `Unknown config pi05_farm_uf850` | The patches didn't apply — re-run `bash setup.sh <token>` and check its verify step passed |
| `HF 401 Unauthorized` | Token expired / wrong scope; re-run setup.sh with a fresh **write** token |
| `squeue --start` shows job pending forever | Out of hour budget — `sshare -u $USER`, message @anthony |
| `No module named openpi` | `cd ~/farm-train/openpi && uv sync` (the setup.sh path should handle this) |

## What if I want to iterate?

The setup is idempotent — re-running `bash setup.sh <token>` re-applies the
patches (no-op if already applied). To change hyperparameters, edit the
`TrainConfig` block in `~/farm-train/patch_openpi_config_pi05.py`, then either
re-clone openpi or hand-edit the inserted block in
`~/farm-train/openpi/src/openpi/training/config.py`, then resubmit
`sbatch train_pi05.sbatch`.

For a longer run, bump `--time` to 24h (max for `small`) or switch partition to
`medium` (5d max) by editing the `#SBATCH` lines.
