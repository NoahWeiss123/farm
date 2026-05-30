#!/bin/bash
# SLURM batch file: FULL fine-tune of π0.5 on the FARM UF850 dataset.
#
# Sized for 8× H100 80GB (the whole node) + 128 CPUs on the CS153 cluster.
# fsdp_devices=2 shards a full fine-tune of π0.5 (~3.3B params) across 2 GPUs
# (it won't fit on one 80GB H100); with 8 GPUs that's 4 data-parallel
# replicas, so the global batch (64, =16/replica) trains at ~4× the single-replica
# throughput. The node has 224 CPUs; we grab 128 to feed a large data-loader
# (num_workers=64) so video decoding can't starve the 8 GPUs.
#
# Walltime padded to 12h so the job never times out — actual training is
# ~1.5-2h (20k steps on 8 GPUs, if the data-loader keeps up), plus ~10-20 min
# for first-time container build + dataset download + norm-stats (norm-stats
# is cached from a prior run, so a resume skips it).
#
# openpi saves a checkpoint every save_interval=5000 steps and (with
# keep_period=5000) retains all of them. The background pusher uploads
# the 4 checkpoints (step 5000, 10000, 15000, 20000) to
# NoahWeiss/farm_uf850_pi05, tagged step-<N>. Pull any:
#
#   hf download NoahWeiss/farm_uf850_pi05 --revision step-19999
#
# Submit:    sbatch train_pi05.sbatch
# Monitor:   squeue -u $USER
#            tail -f train-<jobid>.out
# Cancel:    scancel <jobid>

#SBATCH --job-name=farm-pi05
#SBATCH --partition=small
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=train-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
# HF_TOKEN comes from this file (chmod 600). The dataset is public so the
# LeRobot loader doesn't need it; it's here for the background checkpoint
# pusher, which writes checkpoints to the model repo (writing needs auth
# even though that repo is public).
source "$WORK/.hf_env"
export DATASET_REPO_ID="NoahWeiss/farm_uf850_bottle"
export MODEL_REPO_ID="NoahWeiss/farm_uf850_pi05"
export EXP_NAME="farm_uf850_pi05_${SLURM_JOB_ID}"
export CONFIG_NAME="pi05_farm_uf850"

# ── Resume support ──────────────────────────────────────────────────────
# If a previous run hit walltime, set this to the latest checkpoint dir
# (e.g. checkpoints/pi05_farm_uf850/farm_uf850_pi05_<old_jobid>/19999)
# and resubmit. Leave empty for a fresh run.
RESUME_FROM="${RESUME_FROM:-}"
# Optional step override (the /train dashboard page sets this); empty = config default.
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

# ── Train inside an NGC PyTorch container ───────────────────────────────
# NGC PyTorch ships CUDA + NCCL + cuDNN + ffmpeg matched to host drivers
# AND a JAX-CUDA wheel works on top of it. (openpi is JAX-based.)
#
# --cpus-per-task=128 — feeds the 64 data-loader workers (~2 CPUs each)
# that decode video for the 8 GPUs, plus the first-time enroot squashfs
# build (needs ≥16 to avoid the 30-min single-threaded penalty). The node
# has 224 CPUs; 128 leaves headroom and we have it to ourselves anyway.
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail

# Container runs as root, so \$HOME defaults to /root. Override to the
# mounted user home so all relative-to-\$HOME paths line up with the host.
export HOME='${HOME}'

# Re-export inside the container (srun doesn't always pass user env vars).
export HF_TOKEN='${HF_TOKEN}'
export DATASET_REPO_ID='${DATASET_REPO_ID}'
export MODEL_REPO_ID='${MODEL_REPO_ID}'
export EXP_NAME='${EXP_NAME}'
export RESUME_FROM='${RESUME_FROM}'
export NUM_TRAIN_STEPS='${NUM_TRAIN_STEPS}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Install ffmpeg in the container so torchcodec (LeRobot's video backend)
# can load shared libs. NGC PyTorch doesn't ship system ffmpeg. ~30s.
echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null

# Sync openpi deps (cached on second+ runs).
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

# LeRobot's loader auto-downloads to ~/.cache/huggingface/lerobot/ on
# first access during compute_norm_stats (next step). No separate
# pre-stage needed — saves a duplicate ~796 MB download.

# Verify all 8 GPUs are visible before committing to a multi-hour job.
echo '>>> verifying jax devices…'
uv run python -c \"import jax; ds = jax.devices(); print(f'jax sees {len(ds)} devices: {ds}'); assert len(ds) == 8, f'expected 8 GPUs, got {len(ds)}'\"

# Norm stats (idempotent — fast no-op if checksum matches).
echo '>>> computing norm stats…'
uv run scripts/compute_norm_stats.py --config-name=pi05_farm_uf850

# Launch the background checkpoint pusher BEFORE training so the very
# first save (step 5000) gets uploaded.
CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/pi05_farm_uf850/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo '>>> launching background checkpoint pusher…'
uv run python \$HOME/farm-train/push_checkpoints.py \
    --checkpoint-dir \"\$CKPT_DIR\" \
    --repo-id \"\$MODEL_REPO_ID\" \
    --keep-period 5000 \
    --poll-interval 60 \
    > \$HOME/farm-train/push-\${SLURM_JOB_ID}.out 2>&1 &
PUSHER_PID=\$!
echo \"pusher pid: \$PUSHER_PID\"
trap 'echo \">>> sending SIGTERM to pusher pid \$PUSHER_PID\"; kill -TERM \$PUSHER_PID 2>/dev/null || true; wait \$PUSHER_PID 2>/dev/null || true' EXIT

# Train.
echo '>>> starting training…'
if [[ -n \"\$RESUME_FROM\" ]]; then
  echo \"resuming from \$RESUME_FROM\"
  uv run python scripts/train.py pi05_farm_uf850 \
      --exp-name=\"\$EXP_NAME\" \
      --resume \
      --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}
else
  uv run python scripts/train.py pi05_farm_uf850 \
      --exp-name=\"\$EXP_NAME\" \
      --overwrite \
      --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}
fi

# Drain the pusher: send SIGTERM so it does one final pass picking up
# the very last checkpoint (training is done so the highest-numbered
# step is now safe to upload).
echo '>>> draining checkpoint pusher…'
kill -TERM \$PUSHER_PID 2>/dev/null || true
wait \$PUSHER_PID 2>/dev/null || true
"

echo "✓ training job ${SLURM_JOB_ID} finished"
echo "checkpoints on HF: https://huggingface.co/${MODEL_REPO_ID}"
echo "tags: step-5000, step-10000, step-15000, step-19999 (whichever finished)"
