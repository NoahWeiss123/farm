#!/bin/bash
# Offline action-accuracy + latency eval of a trained π0.5 checkpoint on
# recorder-format episodes. In-process (no serve_policy / websocket). 1× H100.
#
# Parameterized by env (defaults = the GSE fine-tune):
#   EVAL_CONFIG  registered openpi config   (default pi05_farm_uf850_gse)
#   EVAL_REPO    HF repo for the fallback   (default NoahWeiss/farm_uf850_pi05_gse)
#   EVAL_STEP    checkpoint step            (default 2999)
#   EVAL_MODEL   short label for the JSON   (default gse)
#   EVAL_OUT     results JSON path          (default ~/farm-train/eval-<model>.json)
#
# Stage first: model/cluster/eval_offline.py -> ~/farm-train/, a few episode_*
# dirs -> ~/farm-train/eval_episodes/. Submit:
#   sbatch eval_pi05.sbatch                                      # GSE (defaults)
#   EVAL_CONFIG=pi05_farm_uf850 EVAL_REPO=NoahWeiss/farm_uf850_pi05 \
#     EVAL_STEP=19999 EVAL_MODEL=full sbatch eval_pi05.sbatch    # full fine-tune

#SBATCH --job-name=eval-pi05
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/nhweiss/farm-train/eval-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
CONFIG_NAME="${EVAL_CONFIG:-pi05_farm_uf850_gse}"
REPO_ID="${EVAL_REPO:-NoahWeiss/farm_uf850_pi05_gse}"
STEP="${EVAL_STEP:-2999}"
MODEL="${EVAL_MODEL:-gse}"
OUT="${EVAL_OUT:-$WORK/eval-${MODEL}.json}"

# ── Resolve the checkpoint (local training output first; HF as fallback) ──
LOCAL_CKPT=$(ls -d "$WORK"/openpi/checkpoints/${CONFIG_NAME}/*/${STEP} 2>/dev/null | head -1)
STABLE_CKPT="$WORK/checkpoints/${MODEL}_step${STEP}/step-${STEP}"
if [[ -n "$LOCAL_CKPT" && -d "$LOCAL_CKPT/params" ]]; then
  CKPT_DIR="$LOCAL_CKPT"
elif [[ -d "$STABLE_CKPT/params" ]]; then
  CKPT_DIR="$STABLE_CKPT"
else
  echo ">>> no local checkpoint — downloading step-${STEP} from ${REPO_ID}…"
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$REPO_ID" \
      --include "step-${STEP}/*" --local-dir "$WORK/checkpoints/${MODEL}_step${STEP}"
  CKPT_DIR="$STABLE_CKPT"
fi
echo ">>> [${MODEL}] config=${CONFIG_NAME} checkpoint=$CKPT_DIR -> $OUT"

# ── Run the eval inside the openpi container (needs JAX + GPU) ────────────
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

uv run python \$HOME/farm-train/eval_offline.py \
    --config ${CONFIG_NAME} \
    --checkpoint-dir '${CKPT_DIR}' \
    --episodes-dir \$HOME/farm-train/eval_episodes \
    --model ${MODEL} --out '${OUT}' \
    --n-episodes 6 --samples-per-episode 16 --horizon 10
"
echo ">>> [${MODEL}] eval done -> $OUT"
