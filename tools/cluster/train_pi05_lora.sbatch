#!/bin/bash
# SLURM batch file: LoRA fine-tune of π0.5 on the FARM UF850 dataset.
#
# The generalization-preserving alternative to the full-FT train_pi05.sbatch.
# LoRA freezes the ~3.3B pretrained backbone and trains low-rank adapters, so
# π0.5's pretrained priors survive a 2-task transfer (the full FT memorizes the
# two trajectories and erodes them). See patch_openpi_config_pi05_lora.py and
# tools/FINDINGS.md for the why.
#
# Sized for 1× H100 — a π0.5 LoRA fits on one 80GB GPU (no FSDP), which is the
# polite default on the shared cluster. ~10k steps; checkpoints every 2k.
#
# Prereq: setup.sh has registered pi05_farm_uf850_lora (it runs the LoRA patch
# if patch_openpi_config_pi05_lora.py is staged alongside it).
#
# Submit:    sbatch train_pi05_lora.sbatch
# Monitor:   squeue -u $USER ; tail -f train-lora-<jobid>.out
# Pull:      hf download NoahWeiss/farm_uf850_pi05_lora --revision step-9999

#SBATCH --job-name=farm-pi05-lora
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=train-lora-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
export DATASET_REPO_ID="NoahWeiss/farm_uf850_bottle"
export MODEL_REPO_ID="NoahWeiss/farm_uf850_pi05_lora"
export EXP_NAME="farm_uf850_pi05_lora_${SLURM_JOB_ID}"
export CONFIG_NAME="pi05_farm_uf850_lora"
RESUME_FROM="${RESUME_FROM:-}"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export DATASET_REPO_ID='${DATASET_REPO_ID}'
export MODEL_REPO_ID='${MODEL_REPO_ID}'
export EXP_NAME='${EXP_NAME}'
export CONFIG_NAME='${CONFIG_NAME}'
export RESUME_FROM='${RESUME_FROM}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null

pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

echo '>>> verifying jax sees the GPU…'
uv run python -c \"import jax; ds=jax.devices(); print('jax devices:', ds); assert len(ds) >= 1\"

echo '>>> computing norm stats…'
uv run scripts/compute_norm_stats.py --config-name=\$CONFIG_NAME

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo '>>> launching background checkpoint pusher…'
uv run python \$HOME/farm-train/push_checkpoints.py \
    --checkpoint-dir \"\$CKPT_DIR\" \
    --repo-id \"\$MODEL_REPO_ID\" \
    --keep-period 2000 \
    --poll-interval 60 \
    > \$HOME/farm-train/push-lora-\${SLURM_JOB_ID}.out 2>&1 &
PUSHER_PID=\$!
trap 'kill -TERM \$PUSHER_PID 2>/dev/null || true; wait \$PUSHER_PID 2>/dev/null || true' EXIT

echo '>>> starting LoRA training…'
if [[ -n \"\$RESUME_FROM\" ]]; then
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --resume --no-wandb-enabled
else
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled
fi

echo '>>> draining checkpoint pusher…'
kill -TERM \$PUSHER_PID 2>/dev/null || true
wait \$PUSHER_PID 2>/dev/null || true
"

echo "✓ LoRA training job ${SLURM_JOB_ID} finished"
echo "checkpoints on HF: https://huggingface.co/${MODEL_REPO_ID}"
