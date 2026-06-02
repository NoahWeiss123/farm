#!/bin/bash
# Benchmark ONE model's full checkpoint ladder on TRAINING episodes (open-loop
# action-prediction). One job per model; run all in parallel (1 GPU each).
#
# Eval is CLEAN: FARM_AUG_LEVEL / FARM_PROMPT_AUG are deliberately NOT exported
# (they are no-ops unless set), so the policy sees raw frames + canonical task
# prompts — byte-identical to the serve path.
#
# Submit examples (after training completes):
#   GSE: FARM_GSE_NUM_SPECIALIZED=79 EVAL_TAG=e80 \
#        EVAL_CONFIG=pi05_farm_multiobject_gse_sweep \
#        EXP_DIR=$HOME/farm-train/openpi/checkpoints/pi05_farm_multiobject_gse_sweep/farm_gse_sweep_e80_<jobid> \
#        sbatch -J eval-e80 eval_gse_sweep.sbatch
#   FFT: EVAL_TAG=fft EVAL_CONFIG=pi05_farm_multiobject_fft \
#        EXP_DIR=$HOME/farm-train/openpi/checkpoints/pi05_farm_multiobject_fft/farm_fft_multiobject_robust_406 \
#        sbatch -J eval-fft eval_gse_sweep.sbatch

#SBATCH --job-name=eval-gse-sweep
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=eval-gse-sweep-%x-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
: "${EVAL_CONFIG:?set EVAL_CONFIG}"; : "${EVAL_TAG:?set EVAL_TAG}"; : "${EXP_DIR:?set EXP_DIR}"
export EVAL_CONFIG EVAL_TAG EXP_DIR
export REPO_ID="${REPO_ID:-NoahWeiss/farm_uf850_multiobject}"
export RESULTS_DIR="${RESULTS_DIR:-$WORK/eval_gse_sweep/$EVAL_TAG}"
export SEED="${SEED:-0}" NEPS="${NEPS:-80}" SPE="${SPE:-10}" H="${H:-10}" ROLLPT="${ROLLPT:-1}" ROLLSTR="${ROLLSTR:-4}"
# Export the expert count ONLY when set: an empty string is PRESENT (not absent),
# so int(os.environ["FARM_GSE_NUM_SPECIALIZED"]) would raise on int(""). FFT evals
# leave it truly unset (the FFT config never reads it); GSE evals must set it.
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
export EVAL_CONFIG='${EVAL_CONFIG}' EVAL_TAG='${EVAL_TAG}' EXP_DIR='${EXP_DIR}'
export REPO_ID='${REPO_ID}' RESULTS_DIR='${RESULTS_DIR}'
${NS_EXPORT}
export SEED='${SEED}' NEPS='${NEPS}' SPE='${SPE}' H='${H}' ROLLPT='${ROLLPT}' ROLLSTR='${ROLLSTR}'
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# (intentionally NOT exporting FARM_AUG_LEVEL / FARM_PROMPT_AUG — clean eval)

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

bash \$HOME/farm-train/eval_gse_sweep_inner.sh
"

echo "✓ eval-gse-sweep ${EVAL_TAG} job ${SLURM_JOB_ID} done → results in ${RESULTS_DIR}"
