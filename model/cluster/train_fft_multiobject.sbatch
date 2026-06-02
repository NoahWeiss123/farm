#!/bin/bash
# π0.5 FULL fine-tune (the standard PI full-FT pathway, NOT GSE/LoRA) on
# farm_uf850_multiobject (424 eps, 4 tasks: bottle/bear/hat/duck → box),
# made robust with the GSE-era domain-randomization toolkit:
#   FARM_AUG_LEVEL=heavy  — wide brightness/contrast/saturation + hue jitter,
#                           per-channel gamma, grayscale, gaussian blur, crop/rotate
#   FARM_PROMPT_AUG=1      — sample a paraphrase per example across all 4 tasks
#
# 4× H100 data layout: fsdp_devices=2 (config) shards the ~3.3B-param full FT
# across 2 GPUs ⇒ 2 FSDP groups → 2 data-parallel replicas. batch 32 → 8
# samples/GPU for activations (32/4 devices), with params+grads+Adam sharded /2
# — the BYTE-IDENTICAL per-GPU footprint of the proven 8-GPU/batch-64 full FT
# (64/8 = 8/GPU, /2 sharding). 56k steps ≈ 13.9 epochs; permanent checkpoints at
# 8k,16k,24k,32k,40k,48k plus a final-step save at 55999, all streamed to HF by
# the background pusher. Wide CPU (128) + num_workers=64 = 2 CPU/worker so the
# base+wrist h264 decode never starves the GPUs (matches the proven full-FT run).
#
# Submit:  sbatch train_fft_multiobject.sbatch
# Monitor: squeue -u $USER ; tail -f train-fft-multiobj-<jobid>.out
# Model:   https://huggingface.co/NoahWeiss/farm_uf850_multiobject_fft_robust

#SBATCH --job-name=farm-fft-multiobj
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=16:00:00
#SBATCH --output=train-fft-multiobj-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
export MODEL_REPO_ID="NoahWeiss/farm_uf850_multiobject_fft_robust"
export CONFIG_NAME="pi05_farm_multiobject_fft"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

# ── Resume support ──────────────────────────────────────────────────────────
# EXP_NAME keys the checkpoint dir (checkpoints/$CONFIG/$EXP_NAME) and openpi's
# --resume restores ONLY from that dir. A fresh run embeds the job id. To resume
# a walltime-capped run, resubmit with RESUME_FROM=<the OLD EXP_NAME>, e.g.
#   RESUME_FROM=farm_fft_multiobject_robust_1234 sbatch train_fft_multiobject.sbatch
# so we reuse that exact dir and --resume finds the prior checkpoints. (For a
# full-length resume leave NUM_TRAIN_STEPS empty so the original 56k cosine
# schedule — including decay_steps — stays aligned.)
export RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" ]]; then
  export EXP_NAME="$RESUME_FROM"
else
  export EXP_NAME="farm_fft_multiobject_robust_${SLURM_JOB_ID}"
fi

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export MODEL_REPO_ID='${MODEL_REPO_ID}'
export EXP_NAME='${EXP_NAME}'
export CONFIG_NAME='${CONFIG_NAME}'
export NUM_TRAIN_STEPS='${NUM_TRAIN_STEPS}'
export RESUME_FROM='${RESUME_FROM}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Robust full-FT cell: heavy visual domain randomization + prompt paraphrase.
export FARM_AUG_LEVEL=heavy
export FARM_PROMPT_AUG=1

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null

pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

echo '>>> verifying jax sees 4 GPUs…'
uv run python -c \"import jax; ds=jax.devices(); print('jax devices:', ds); assert len(ds) >= 4, f'want 4 GPUs, got {len(ds)}'\"

echo \">>> aug=\$FARM_AUG_LEVEL prompt_aug=\$FARM_PROMPT_AUG config=\$CONFIG_NAME\"
echo '>>> computing norm stats (one-time, 129k frames)…'
uv run scripts/compute_norm_stats.py --config-name=\$CONFIG_NAME

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo \">>> checkpoints → \$CKPT_DIR  (streamed to \$MODEL_REPO_ID)\"
uv run python \$HOME/farm-train/push_checkpoints.py \
    --checkpoint-dir \"\$CKPT_DIR\" \
    --repo-id \"\$MODEL_REPO_ID\" \
    --keep-period 8000 \
    --poll-interval 60 \
    > \$HOME/farm-train/push-fft-multiobj-\${SLURM_JOB_ID}.out 2>&1 &
PUSHER_PID=\$!
trap 'kill -TERM \$PUSHER_PID 2>/dev/null || true; wait \$PUSHER_PID 2>/dev/null || true' EXIT

echo '>>> starting FULL fine-tune (4 GPU, batch 32, 56k steps, heavy aug + prompt aug)…'
if [[ -n \"\$RESUME_FROM\" ]]; then
  echo \"resuming from \$RESUME_FROM\"
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --resume --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}
else
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}
fi

echo '>>> draining checkpoint pusher…'
kill -TERM \$PUSHER_PID 2>/dev/null || true
wait \$PUSHER_PID 2>/dev/null || true
echo '>>> final checkpoints:'
ls -1 \"\$CKPT_DIR\" 2>/dev/null || true
"

echo "✓ FFT-multiobject job ${SLURM_JOB_ID} finished — GPUs released"
echo "model: https://huggingface.co/${MODEL_REPO_ID}"
