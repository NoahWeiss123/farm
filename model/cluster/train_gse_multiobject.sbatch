#!/bin/bash
# Flagship π0.5 GSE-robust fine-tune on farm_uf850_multiobject (424 eps, 4 tasks).
#
# The "robust" cell of the GSE ablation: heavy domain-randomization aug
# (FARM_AUG_LEVEL=heavy) + prompt paraphrase (FARM_PROMPT_AUG=1) on the
# SVD-init GSE recipe. 6× H100 data-parallel (no FSDP — frozen backbone fits one
# GPU, the 6 replicate). batch 192 (32/GPU), 6k steps, checkpoints every 1k
# streamed to HF. Wide CPU + num_workers=96 so the video-decode loader feeds all
# 6 GPUs and the one-time norm-stats pass over 129k frames is fast.
#
# Submit:  sbatch train_gse_multiobject.sbatch
# Monitor: squeue -u $USER ; tail -f train-gse-multiobj-<jobid>.out
# Model:   https://huggingface.co/NoahWeiss/farm_uf850_multiobject_gse_robust

#SBATCH --job-name=farm-gse-multiobj
#SBATCH --partition=small
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=144
#SBATCH --time=08:00:00
#SBATCH --output=train-gse-multiobj-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
export MODEL_REPO_ID="NoahWeiss/farm_uf850_multiobject_gse_robust"
export EXP_NAME="farm_gse_multiobject_robust_${SLURM_JOB_ID}"
export CONFIG_NAME="pi05_farm_multiobject_gse"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

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
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Flagship 'robust' cell: heavy visual domain randomization + prompt paraphrase.
export FARM_AUG_LEVEL=heavy
export FARM_PROMPT_AUG=1

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null

pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

echo '>>> verifying jax sees 6 GPUs…'
uv run python -c \"import jax; ds=jax.devices(); print('jax devices:', ds); assert len(ds) >= 6, f'want 6 GPUs, got {len(ds)}'\"

echo \">>> aug=\$FARM_AUG_LEVEL prompt_aug=\$FARM_PROMPT_AUG config=\$CONFIG_NAME\"
echo '>>> computing norm stats (one-time, 129k frames)…'
uv run scripts/compute_norm_stats.py --config-name=\$CONFIG_NAME

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo \">>> checkpoints → \$CKPT_DIR  (streamed to \$MODEL_REPO_ID)\"
uv run python \$HOME/farm-train/push_checkpoints.py \
    --checkpoint-dir \"\$CKPT_DIR\" \
    --repo-id \"\$MODEL_REPO_ID\" \
    --keep-period 1000 \
    --poll-interval 60 \
    > \$HOME/farm-train/push-gse-multiobj-\${SLURM_JOB_ID}.out 2>&1 &
PUSHER_PID=\$!
trap 'kill -TERM \$PUSHER_PID 2>/dev/null || true; wait \$PUSHER_PID 2>/dev/null || true' EXIT

echo '>>> starting GSE-robust training (6 GPU, batch 192, 6k steps)…'
uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}

echo '>>> draining checkpoint pusher…'
kill -TERM \$PUSHER_PID 2>/dev/null || true
wait \$PUSHER_PID 2>/dev/null || true
echo '>>> final checkpoints:'
ls -1 \"\$CKPT_DIR\" 2>/dev/null || true
"

echo "✓ GSE-multiobject job ${SLURM_JOB_ID} finished — GPUs released"
echo "model: https://huggingface.co/${MODEL_REPO_ID}"
