#!/bin/bash
# π0.5 LoRA fine-tune on the 100-episode bottle dataset, INITIALIZED OFF THE
# FINE-TUNED GSE MULTIOBJECT CHECKPOINT (step-5999) instead of base π0.5.
#
# Same recipe as train_pi05_lora_bottle100.sbatch; the only difference is the
# config's weight_loader = GSEMergeWeightLoader (merges the GSE adapters into the
# dense backbone at load, so the LoRA trains on top of the multiobject policy).
#
# 1× H100 (LoRA fits one 80GB GPU, no FSDP) with a wide CPU allocation +
# num_workers=32 so the on-the-fly mp4 decode never starves the GPU.
#
# Submit:   sbatch train_pi05_lora_bottle100_gse.sbatch
# Monitor:  squeue -u $USER ; tail -f train-bottlelora-gse-<jobid>.out
# Ckpts:    ~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora_gse/<exp>/<step>/

#SBATCH --job-name=farm-bottlelora-gse
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=train-bottlelora-gse-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
export EXP_NAME="pi05_farm_bottle_lora_gse_${SLURM_JOB_ID}"
export CONFIG_NAME="pi05_farm_bottle_lora_gse"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export EXP_NAME='${EXP_NAME}'
export CONFIG_NAME='${CONFIG_NAME}'
export NUM_TRAIN_STEPS='${NUM_TRAIN_STEPS}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENPI_CONFIG_PY=\"\$HOME/farm-train/openpi/src/openpi/training/config.py\"

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null

pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

echo '>>> applying GSE-merge weight loader + LoRA-off-GSE config patches (idempotent)…'
uv run python \"\$HOME/farm-train/patch_openpi_gse_merge.py\" --openpi-root \"\$HOME/farm-train/openpi\"
uv run python \"\$HOME/farm-train/patch_openpi_config_pi05_lora_bottle100_gse.py\"

echo '>>> verifying jax sees the GPU…'
uv run python -c \"import jax; ds=jax.devices(); print('jax devices:', ds); assert len(ds) >= 1\"

echo '>>> verifying the config + GSE init params resolve…'
uv run python -c \"
import openpi.training.config as C
cfg = C.get_config('\$CONFIG_NAME')
wl = cfg.weight_loader
print('config:', cfg.name, '| weight_loader:', type(wl).__name__)
import os
p = wl.params_path
print('GSE init params:', p, '| exists:', os.path.isdir(p))
assert os.path.isdir(p), 'GSE init params dir missing: ' + p
\"

echo '>>> computing norm stats…'
uv run scripts/compute_norm_stats.py --config-name=\$CONFIG_NAME

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo \">>> checkpoints will land in: \$CKPT_DIR\"

echo '>>> starting LoRA-off-GSE training (the GSE adapters merge into the dense base at weight load)…'
uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}

echo '>>> training finished; final checkpoints:'
ls -1 \"\$CKPT_DIR\" 2>/dev/null || true
"

echo "✓ bottle-LoRA-off-GSE training job ${SLURM_JOB_ID} finished — GPU released"
echo "checkpoints: ~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora_gse/${EXP_NAME}/"
