# Fine-tuning π0.5 on the CS153 H100 cluster

End-to-end workflow for training the FARM UF850 bottle policy on the cluster,
with **three interchangeable fine-tuning architectures** you can run and
compare. Assumes:

* Dataset on the Hub at **`NoahWeiss/farm_uf850_bottle`** (200 episodes, 2 tasks,
  59,183 frames — see `model/HUGGINGFACE_UPLOAD.md` to push it).
* openpi cloned + the FARM configs registered by the patch scripts here
  (`setup.sh` does this, step 3).
* Cluster access via the Omniva kubeconfig (see `.claude/CLAUDE.md`).

## The three architectures

All three fine-tune the same `pi05_base` checkpoint on the same dataset, output
`action_horizon=10` absolute joint targets, and serve/eval through the same
path (`serve_pi05.sbatch` + `model/eval_pi05.py`) — so they are directly
comparable. They differ only in *which* parameters adapt and *how*:

| Config | What trains | GPUs | Why |
|---|---|---|---|
| `pi05_farm_uf850` (full FT) | all ~3.3B params | 8 (FSDP×2 · DP×4) | max capacity; **overfits** 2-task data + erodes the base |
| `pi05_farm_uf850_lora` | LoRA adapters + action expert | 1 | preserves the base; **under-adapts** when precise control is needed |
| `pi05_farm_uf850_gse` | SVD experts + action expert | 1 | **VLA-GSE** (arXiv:2605.06175): preserves the dominant subspace (generalized expert) *and* adapts residual subspaces (specialized experts) — best of both |

See `model/FINDINGS.md` for the full diagnosis of why full FT alone
underperforms here, and `openpi_gse.py` for the GSE method. The patch scripts
register all three; the default sbatch trains full FT, and there are dedicated
1-GPU sbatches for LoRA and GSE.

## 1. Shell into the login pod (from your laptop)

```bash
USER_NAME=<your-cluster-username>   # your Omniva one, e.g. nhweiss
POD=$(kubectl get pod -n slurm -l stanford/user=${USER_NAME} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it -n slurm $POD -c login -- runuser -l ${USER_NAME}
```

## 2. Stage the launcher files to the pod

From your **laptop**, in another terminal. Stage everything under
`~/farm-train` on the pod — the data-config patch, all three config patches,
the GSE module + wiring patch, the three sbatches, and the checkpoint pusher:

```bash
USER_NAME=<your-cluster-username>
POD=$(kubectl get pod -n slurm -l stanford/user=${USER_NAME} -o jsonpath='{.items[0].metadata.name}')
DST=slurm/$POD:/home/$USER_NAME/farm-train
kubectl exec -n slurm $POD -c login -- runuser -l ${USER_NAME} -c "mkdir -p ~/farm-train"
for f in setup.sh push_checkpoints.py \
         patch_openpi_config.py \
         patch_openpi_config_pi05.py patch_openpi_config_pi05_lora.py \
         openpi_gse.py patch_openpi_gse.py patch_openpi_config_pi05_gse.py \
         train_pi05.sbatch train_pi05_lora.sbatch train_pi05_gse.sbatch \
         serve_pi05.sbatch; do
  kubectl cp model/cluster/$f $DST/$f -c login
done
```

## 3. Run setup (on the login pod)

```bash
cd ~/farm-train
bash setup.sh <PASTE_YOUR_HF_TOKEN_HERE>
```

This clones openpi, then idempotently patches `src/openpi/training/config.py`
(openpi reads its registry from the `_CONFIGS` list there — no config package to
drop a file into) to register **all three** configs + the shared
`LeRobotFarmDataConfig`, installs the GSE module + `GSESVDWeightLoader`, verifies
everything py-compiles, and writes a chmod-600 `.hf_env` with your token. The
verify step prints which configs registered.

## 4. Train

Pick an architecture. Each streams checkpoints to its own HF repo during the run
(`step-<N>` tags) via the background pusher.

```bash
# Full fine-tune — 8 GPUs (FSDP×2 + DP×4), batch 64, 20k steps (~2-3.5h),
# → NoahWeiss/farm_uf850_pi05
sbatch train_pi05.sbatch

# LoRA — 1 GPU, batch 32, 12k steps (~5-7h) → NoahWeiss/farm_uf850_pi05_lora
sbatch train_pi05_lora.sbatch

# GSE — 1 GPU, batch 32, 12k steps (~5-7h, ~6 GPU-hours) → NoahWeiss/farm_uf850_pi05_gse
#   ⚠ Smoke-test first (see model/FINDINGS.md) — GSE is syntax+math-validated
#   but not yet GPU-tested. For ~2-4x faster wall-clock bump --gres to gpu:2/4
#   (it data-parallel-replicates; no FSDP).
sbatch train_pi05_gse.sbatch
```

**GPU-hours, not just wall-clock:** full FT burns ~28 H100-hours (8 GPUs × ~3.5h);
LoRA and GSE freeze the backbone (no Adam state for the ~3.3B params, no FSDP),
fit on **one** GPU, and cost ~6 H100-hours — **~4-5× cheaper**, and they leave the
8-GPU node free. GSE's SVD-init also converges in fewer steps than random-init
LoRA (its "faster"); both use the same 12k-step budget so the comparison is fair.
For full FT, the 8 GPUs are a *memory* requirement: π0.5 (~3.3B) + Adam state
(~40GB) doesn't fit one 80GB H100, so `fsdp_devices=2` shards it and the node's
other 6 GPUs add 3 data-parallel replicas (same final weights, ~4× throughput).

## 5. Monitor

```bash
squeue -u $USER                            # is it running?
tail -f ~/farm-train/train-<jobid>.out     # training logs (train-lora-/train-gse- for the others)
tail -f ~/farm-train/push-<jobid>.out      # checkpoint-pusher logs
sacct -u $USER -S today                    # job history
sshare -u $USER                            # fairshare / usage
```

## 6. After training

Checkpoints stream to the per-architecture HF repo, tagged `step-<N>`
(full FT: 5000/10000/15000/19999 — openpi names the final at step N-1; LoRA/GSE:
every 2000). **Select by held-out performance, not the last step** — earlier
checkpoints often generalize better (see `model/FINDINGS.md`). Pull any:

```bash
hf download NoahWeiss/farm_uf850_pi05 --include 'step-19999/*' --local-dir ~/farm_pi05_step19999
```

To serve + run on the arm, see `DEPLOYMENT.md`. The serve sbatch + `model/eval_pi05.py`
work unchanged for all three (same obs/action contract).

## Iterating on hyperparameters

`setup.sh` is idempotent (re-running re-applies patches as no-ops). To change a
config, edit the relevant `patch_openpi_config_pi05*.py` block, then either
re-clone openpi or hand-edit the inserted `TrainConfig` in
`~/farm-train/openpi/src/openpi/training/config.py`, and resubmit. For a longer
run bump `--time` (max 24h on `small`, 5d on `medium`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pyxis: importing` hangs >5 min on first run | First-time squashfs build of the 20GB PyTorch image — wait ~3 min with `--cpus-per-task=16`+ |
| `JSON parse error` in container init | You used `docker.io#library/<name>`; switch to `nvcr.io#…` or the bare name |
| `CUDA out of memory` (full FT) | Drop `batch_size` 64→32 in `patch_openpi_config_pi05.py`, re-run setup.sh |
| `Unknown config pi05_farm_uf850*` | Patches didn't apply — re-run `bash setup.sh <token>`, check the verify step |
| GSE: shape/JIT error on first step | Run the smoke test in `model/FINDINGS.md`; GSE's integration is validated for syntax/math but not yet on GPU |
| `HF 401 Unauthorized` | Token expired / wrong scope; re-run setup.sh with a fresh **write** token |
| Job pending forever | Out of fairshare/budget — `sshare -u $USER` |
