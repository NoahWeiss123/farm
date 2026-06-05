#!/bin/bash
# eval_curve.sbatch — error-vs-training-step curve for ONE model (both splits).
# One job per model; run all 5 in parallel (1 GPU each). Wraps eval_curve_inner.sh.
#
# CLEAN eval: FARM_AUG_LEVEL / FARM_PROMPT_AUG deliberately NOT exported.
#
# Submit (after a model's training completes):
#   GSE (local checkpoints):
#     FARM_GSE_NUM_SPECIALIZED=7 MODEL_TAG=e8 EVAL_CONFIG=pi05_farm_multiobject_gse_sweep \
#       SOURCE_KIND=local \
#       EXP_DIR=$HOME/farm-train/openpi/checkpoints/pi05_farm_multiobject_gse_sweep/farm_gse_sweep_e8_730 \
#       sbatch -J evalc-e8 eval_curve.sbatch
#   FFT (download tags from HF):
#     MODEL_TAG=fft EVAL_CONFIG=pi05_farm_multiobject_fft SOURCE_KIND=hf \
#       HF_REPO=NoahWeiss/farm_uf850_multiobject_fft_robust \
#       sbatch -J evalc-fft eval_curve.sbatch

#SBATCH --job-name=eval-curve
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=eval-curve-%x-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
: "${MODEL_TAG:?set MODEL_TAG}"; : "${EVAL_CONFIG:?set EVAL_CONFIG}"; : "${SOURCE_KIND:?set SOURCE_KIND=local|hf}"
export MODEL_TAG EVAL_CONFIG SOURCE_KIND
export EXP_DIR="${EXP_DIR:-}" HF_REPO="${HF_REPO:-}"
export RESULTS_DIR="${RESULTS_DIR:-$WORK/eval_curve/$MODEL_TAG}"
export STEPS="${STEPS:-8000 16000 24000 32000 40000 48000 55999}"
export TRAIN_RANGES="${TRAIN_RANGES:-0:239,299:339,349:374,384:414}"
export HELDOUT_RANGES="${HELDOUT_RANGES:-239:299,339:349,374:384,414:424}"
export N_EPISODES="${N_EPISODES:-80}" SAMPLES="${SAMPLES:-10}" SEED="${SEED:-0}"

# Export expert count ONLY when set (empty string would break int("") in the GSE config).
FARM_GSE_NUM_SPECIALIZED="${FARM_GSE_NUM_SPECIALIZED:-}"
if [[ -n "$FARM_GSE_NUM_SPECIALIZED" ]]; then
  NS_EXPORT="export FARM_GSE_NUM_SPECIALIZED='$FARM_GSE_NUM_SPECIALIZED'"
else
  NS_EXPORT="unset FARM_GSE_NUM_SPECIALIZED 2>/dev/null || true"
fi

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export MODEL_TAG='${MODEL_TAG}' EVAL_CONFIG='${EVAL_CONFIG}' SOURCE_KIND='${SOURCE_KIND}'
export EXP_DIR='${EXP_DIR}' HF_REPO='${HF_REPO}' RESULTS_DIR='${RESULTS_DIR}'
export STEPS='${STEPS}' TRAIN_RANGES='${TRAIN_RANGES}' HELDOUT_RANGES='${HELDOUT_RANGES}'
export N_EPISODES='${N_EPISODES}' SAMPLES='${SAMPLES}' SEED='${SEED}'
${NS_EXPORT}
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# NOTE: deliberately NOT enabling HF_HUB_ENABLE_HF_TRANSFER — on this cluster's
# flaky Hub connection it silently writes TRUNCATED checkpoint shards (orbax then
# fails with 'Truncated Zstd-compressed stream'). The standard downloader verifies.

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

bash \$HOME/farm-train/eval_curve_inner.sh
"

echo "✓ eval-curve ${MODEL_TAG} job ${SLURM_JOB_ID} done → ${RESULTS_DIR:-$WORK/eval_curve/$MODEL_TAG}"
