#!/bin/bash
# Train ONE per-task LoRA on top of the full-FT multi-object model (FFT-56k).
# Init off FFT-56k (CheckpointWeightLoader in the pi05_fftlora config), ONLY the
# LoRA adapters trainable (config freezes all non-lora params), NO aug
# (FARM_AUG_LEVEL=off + FARM_PROMPT_AUG unset → one fixed prompt, unshifted
# frames), task selected by FARM_EP_RANGE (a frame Subset of the full set).
# Identical config/seed for every task so the adapters are directly comparable as
# vectors. 1× H100. Checkpoints stream to a per-task HF repo.
#
#   sbatch train_fftlora.sbatch <task>
#   task ∈ bottle30 bottle100 bear30 duck30 hat30 bottle30s1   (s1 = seed-1 control)
#   tail -f train-fftlora-<jobid>.out
#
# Equal-size n=30 vector set {bottle30,bear30,duck30,hat30} (seed 42) → clean
# vector comparison; each leaves a held-out tail (bottle 30:299, bear 329:349,
# duck 379:384, hat 414:424) for the base-vs-LoRA eval. bottle100 (= user's "first
# 100") is the headline-eval LoRA + a data-scaling point vs bottle30. bottle30s1
# (seed 1) is the same-task/different-init control for the cosine baseline.
#
# Shared norm stats: copies the FFT base model's full-dataset norm_stats.json into
# this config's asset dir so every task-LoRA normalizes inputs identically to the
# FFT base (the LoRA delta is the only variable). HARD-FAILS if missing.

#SBATCH --job-name=fftlora
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=train-fftlora-%j.out
# Modest --mem/--cpus (not the 960G/64 default) so several LoRAs BACKFILL onto the
# spare GPUs of partially-used nodes — a frozen-base single-task LoRA needs little
# host RAM, and this lets the 6 run concurrently on a busy shared cluster.

set -euo pipefail
TASK="${1:?usage: sbatch train_fftlora.sbatch <bottle30|bottle100|bear30|duck30|hat30|bottle30s1>}"
WORK="$HOME/farm-train"
source "$WORK/.hf_env"

# task → (episode range, seed). Tasks contiguous: bottle 0-298, bear 299-348,
# duck 349-383, hat 384-423. n=30 equal-size set + bottle100 + a seed-1 control.
SEED=42
case "$TASK" in
  bottle30)   EPR="0:30"      ;;
  bottle100)  EPR="0:100"     ;;
  bear30)     EPR="299:329"   ;;
  duck30)     EPR="349:379"   ;;
  hat30)      EPR="384:414"   ;;
  bottle30s1) EPR="0:30"; SEED=1 ;;
  *) echo "unknown task '$TASK'"; exit 2 ;;
esac
export FARM_EP_RANGE="$EPR"
export SEED
export CONFIG_NAME="pi05_fftlora"
export EXP_NAME="fftlora_${TASK}"
export MODEL_REPO_ID="NoahWeiss/farm_fftlora_${TASK}"

echo ">>> LoRA task=$TASK  episodes=$EPR  seed=$SEED  repo=$MODEL_REPO_ID"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export CONFIG_NAME='${CONFIG_NAME}'
export EXP_NAME='${EXP_NAME}'
export MODEL_REPO_ID='${MODEL_REPO_ID}'
export FARM_EP_RANGE='${FARM_EP_RANGE}'
export SEED='${SEED}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# NO augmentation: one fixed prompt per task (FARM_PROMPT_AUG unset) + unshifted
# frames (FARM_AUG_LEVEL=off). Explicit so it can't fall back to openpi's mild
# default crop/jitter.
export FARM_AUG_LEVEL=off
unset FARM_PROMPT_AUG

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

echo '>>> verifying 1 GPU…'
uv run python -c \"import jax; print('jax devices:', jax.devices())\"

# SHARED norm stats: copy the FFT base model's full-dataset stats into this
# config's asset dir so every task-LoRA normalizes identically to the FFT base.
# HARD-FAIL if missing — a per-task recompute would contaminate the comparison.
SRC_STATS=\"\$HOME/farm-train/openpi/assets/pi05_farm_multiobject_fft/NoahWeiss/farm_uf850_multiobject/norm_stats.json\"
DST_DIR=\"\$HOME/farm-train/openpi/assets/\$CONFIG_NAME/NoahWeiss/farm_uf850_multiobject\"
[[ -f \"\$SRC_STATS\" ]] || { echo \"FATAL: FFT norm stats missing at \$SRC_STATS — comparison would be contaminated\"; exit 1; }
mkdir -p \"\$DST_DIR\"
cp -f \"\$SRC_STATS\" \"\$DST_DIR/norm_stats.json\"
echo \">>> shared norm stats placed (sha256 \$(sha256sum \"\$DST_DIR/norm_stats.json\" | cut -c1-16)…)\"

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo \">>> checkpoints → \$CKPT_DIR (streamed to \$MODEL_REPO_ID)\"
uv run python \$HOME/farm-train/push_checkpoints.py \
    --checkpoint-dir \"\$CKPT_DIR\" \
    --repo-id \"\$MODEL_REPO_ID\" \
    --keep-period 3000 \
    --poll-interval 60 \
    > \$HOME/farm-train/push-fftlora-\${EXP_NAME}-\${SLURM_JOB_ID}.out 2>&1 &
PUSHER_PID=\$!
trap 'kill -TERM \$PUSHER_PID 2>/dev/null || true; wait \$PUSHER_PID 2>/dev/null || true' EXIT

echo \">>> training LoRA (task=\$EXP_NAME, FARM_EP_RANGE=\$FARM_EP_RANGE, seed=\$SEED, aug=off, adapters-only)…\"
uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --seed=\$SEED --overwrite --no-wandb-enabled

echo '>>> draining checkpoint pusher…'
kill -TERM \$PUSHER_PID 2>/dev/null || true
wait \$PUSHER_PID 2>/dev/null || true
echo '>>> final checkpoints:'; ls -1 \"\$CKPT_DIR\" 2>/dev/null || true
"

echo "✓ fftlora ${TASK} job ${SLURM_JOB_ID} finished — model: https://huggingface.co/${MODEL_REPO_ID}"
