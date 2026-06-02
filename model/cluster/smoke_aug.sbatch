#!/bin/bash
# 1-GPU, 5-step smoke test of the heavy domain-randomization + prompt-aug path,
# run BEFORE committing the 4-GPU overnight job. Validates end-to-end that:
#   * the patched model.py "heavy" augmax chain (hue/gamma/grayscale/blur/crop)
#     builds and runs under jax.vmap inside compute_loss,
#   * the env-gated PromptParaphrase transform wires into the data pipeline,
#   * GSE SVD-init still loads the base weights, and
#   * 5 optimisation steps log a finite, decreasing loss.
# ~8-12 min incl. first-time container squashfs build.
#
#   sbatch smoke_aug.sbatch ; tail -f smoke-aug-<jobid>.out   # expect ">>> SMOKE_AUG_OK"

#SBATCH --job-name=farm-smoke-aug
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:45:00
#SBATCH --output=smoke-aug-%j.out

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
export FARM_AUG_LEVEL=heavy
export FARM_PROMPT_AUG=1
echo '>>> installing ffmpeg…'
apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen
echo '>>> jax devices:'; uv run python -c \"import jax; print(jax.devices())\"
# Compute norm stats fresh (mirrors the overnight job exactly) so this smoke is a
# faithful GO/NO-GO for it — rules out a stale/missing-stats NaN.
echo '>>> norm stats…'; uv run scripts/compute_norm_stats.py --config-name=pi05_farm_uf850_gse
echo '>>> 20-step smoke with FARM_AUG_LEVEL=heavy FARM_PROMPT_AUG=1 (watch for finite, decreasing loss)…'
uv run python scripts/train.py pi05_farm_uf850_gse --exp-name=gse_smoke_aug \
    --overwrite --no-wandb-enabled --num-train-steps=20
echo '>>> SMOKE_AUG_OK'
"
echo "smoke job ${SLURM_JOB_ID} finished"
