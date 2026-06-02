#!/bin/bash
# 1-GPU, ~12-step smoke of the FFT-LoRA training path, run AFTER the FFT-56k run
# finishes (needs .../farm_fft_multiobject_robust_406/55999/params). Uses bear30
# (FARM_EP_RANGE=299:329, a NON-zero-start range) to specifically exercise the
# frame-Subset episode filter, and validates end-to-end:
#   * the new pi05_fftlora config builds (freeze-all-but-lora filter evaluates),
#   * CheckpointWeightLoader inits the frozen base off FFT-56k/55999,
#   * ONLY the LoRA adapters are trainable (prints the trainable param count),
#   * FARM_EP_RANGE=299:329 builds an 8238-frame Subset with NO IndexError,
#   * shared norm stats are found, FARM_AUG_LEVEL=off, and
#   * a few steps log a finite, decreasing loss.
# No pusher; cleans up its own checkpoint dir.
#
#   sbatch smoke_fftlora.sbatch ; tail -f smoke-fftlora-<jobid>.out
#   expect ">>> SMOKE_FFTLORA_OK"

#SBATCH --job-name=smoke-fftlora
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=smoke-fftlora-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
source "$WORK/.hf_env"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export FARM_EP_RANGE=299:329
export FARM_AUG_LEVEL=off
unset FARM_PROMPT_AUG
CONFIG=pi05_fftlora

FFT55=\"\$HOME/farm-train/openpi/checkpoints/pi05_farm_multiobject_fft/farm_fft_multiobject_robust_406/lora_base/params\"
[[ -d \"\$FFT55\" ]] || { echo \"FATAL: FFT base \$FFT55 missing — set the lora_base symlink (post-selection) before the smoke\"; exit 1; }

echo '>>> ffmpeg…'; apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv; cd \"\$HOME/farm-train/openpi\"; uv sync --frozen
echo '>>> jax devices:'; uv run python -c \"import jax; print(jax.devices())\"

# shared norm stats (must exist)
SRC=\"\$HOME/farm-train/openpi/assets/pi05_farm_multiobject_fft/NoahWeiss/farm_uf850_multiobject/norm_stats.json\"
DST=\"\$HOME/farm-train/openpi/assets/\$CONFIG/NoahWeiss/farm_uf850_multiobject\"
[[ -f \"\$SRC\" ]] || { echo 'FATAL: FFT norm stats missing'; exit 1; }
mkdir -p \"\$DST\"; cp -f \"\$SRC\" \"\$DST/norm_stats.json\"

echo '>>> 12-step LoRA smoke (bear30, init off FFT-56k, adapters-only, aug=off)…'
uv run python scripts/train.py \$CONFIG --exp-name=fftlora_smoke --overwrite --no-wandb-enabled --num-train-steps=12
echo '>>> SMOKE_FFTLORA_OK'
echo '>>> cleaning smoke checkpoint dir…'
rm -rf \"\$HOME/farm-train/openpi/checkpoints/\$CONFIG/fftlora_smoke\"
"
echo "smoke-fftlora job ${SLURM_JOB_ID} finished"
