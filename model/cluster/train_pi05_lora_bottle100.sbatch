#!/bin/bash
# SLURM batch file: π0.5 LoRA fine-tune on farm_bottle_lora (100 episodes).
#
# Single-task (bottle→box), 100-episode run on the NEW NoahWeiss/farm_bottle_lora
# dataset (first 100 of FARM dataset4). LoRA freezes the ~3.3B π0.5 backbone and
# trains low-rank adapters, so the pretrained priors survive — see
# patch_openpi_config_pi05_lora_bottle100.py and model/FINDINGS.md.
#
# Sized for 1× H100 (LoRA fits one 80GB GPU, no FSDP) with a deliberately wide
# CPU allocation + num_workers=32 so the on-the-fly video decode never starves
# the GPU. ~10k steps (~12 epochs); checkpoints every 2k to the cluster's
# persistent home (NOT pushed to HF — keep them local; push the best later).
#
# Submit:   sbatch train_pi05_lora_bottle100.sbatch
# Monitor:  squeue -u $USER ; tail -f train-bottlelora-<jobid>.out
# Ckpts:    ~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora/<exp>/<step>/

#SBATCH --job-name=farm-bottle-lora
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=train-bottlelora-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
export DATASET_REPO_ID="NoahWeiss/farm_bottle_lora"
export EXP_NAME="pi05_farm_bottle_lora_${SLURM_JOB_ID}"
export CONFIG_NAME="pi05_farm_bottle_lora"
# Optional step override; empty = config default (10_000).
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export DATASET_REPO_ID='${DATASET_REPO_ID}'
export EXP_NAME='${EXP_NAME}'
export CONFIG_NAME='${CONFIG_NAME}'
export NUM_TRAIN_STEPS='${NUM_TRAIN_STEPS}'
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
echo \">>> checkpoints will land in: \$CKPT_DIR\"

echo '>>> starting LoRA training…'
uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}

echo '>>> training finished; final checkpoints:'
ls -1 \"\$CKPT_DIR\" 2>/dev/null || true
"

echo "✓ bottle-LoRA training job ${SLURM_JOB_ID} finished — GPU released"
echo "checkpoints (cluster, persistent home): ~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora/${EXP_NAME}/"
