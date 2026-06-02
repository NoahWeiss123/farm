#!/bin/bash
# Generic real-episode end-of-horizon eval for the multiobject GSE policy on an
# ARBITRARY dataset (in-distribution OR out-of-distribution). Same model, same
# eval_train_endhorizon.py; only REPO_ID + the output TAG change, so an OOD run
# can sit next to the in-dist run without clobbering it.
#
#   TAG=ood REPO_ID=NoahWeiss/farm_uf850_bottle  -> eval-eh-ood.json + -raw.npz
set -uo pipefail
WORK="$(cd "$(dirname "$0")" && pwd)"
export HOME="$(dirname "$WORK")"
[ -f "$WORK/.hf_env" ] && source "$WORK/.hf_env"
export HF_HUB_ENABLE_HF_TRANSFER=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
unset FARM_AUG_LEVEL FARM_PROMPT_AUG          # inference: no augmentation / paraphrase

TAG="${TAG:-ood}"
CONFIG="${CONFIG:-pi05_farm_multiobject_gse}"
REPO_ID="${REPO_ID:-NoahWeiss/farm_uf850_bottle}"
N_EPISODES="${N_EPISODES:-64}"
SAMPLES="${SAMPLES:-12}"
HORIZON="${HORIZON:-10}"
ROLL_PER_TASK="${ROLL_PER_TASK:-2}"
ROLL_STRIDE="${ROLL_STRIDE:-2}"
# The MODEL under test is the multiobject GSE checkpoint, regardless of dataset.
PARENT="openpi/checkpoints/pi05_farm_multiobject_gse/farm_gse_multiobject_robust_190"
FALLBACK_REPO="NoahWeiss/farm_uf850_multiobject_gse_robust"
FALLBACK_STEP="5999"

highest_step() {  # path of the highest-numbered step dir under $1 (depth<=2)
  find "$1" -maxdepth 2 -type d -regex '.*/[0-9]+$' 2>/dev/null \
    | awk -F/ '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2-
}

set -e
echo ">>> ffmpeg (lerobot video decode)…"
apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv; cd "$WORK/openpi"; uv sync --frozen
set +e

CKPT="$(highest_step "$WORK/$PARENT")"
if [ -z "$CKPT" ] || [ ! -d "$CKPT/params" ]; then
  echo ">>> no local checkpoint; downloading step-$FALLBACK_STEP from $FALLBACK_REPO…"
  DL="$WORK/checkpoints/evaltrain_multiobject"
  hf download "$FALLBACK_REPO" --include "step-${FALLBACK_STEP}/*" --local-dir "$DL" >/dev/null 2>&1
  CKPT="$DL/step-${FALLBACK_STEP}"
fi
[ -d "$CKPT/params" ] || { echo "!! no checkpoint resolved ($CKPT)"; exit 1; }

OUT_JSON="$WORK/eval-eh-${TAG}.json"
OUT_NPZ="$WORK/eval-eh-${TAG}-raw.npz"
echo "================ REAL-EPISODE PREDICTION EVAL · TAG=$TAG ================"
echo "  MODEL: $CONFIG  ckpt=$CKPT"
echo "  DATA : $REPO_ID  (n_episodes=$N_EPISODES samples/ep=$SAMPLES horizon=$HORIZON)"
echo "  rollout: $ROLL_PER_TASK whole eps/task, stride $ROLL_STRIDE"
uv run python "$WORK/eval_train_endhorizon.py" \
    --config "$CONFIG" --checkpoint-dir "$CKPT" --repo-id "$REPO_ID" \
    --n-episodes "$N_EPISODES" --samples-per-episode "$SAMPLES" --horizon "$HORIZON" \
    --roll-per-task "$ROLL_PER_TASK" --roll-stride "$ROLL_STRIDE" \
    --model "$TAG" --out "$OUT_JSON" --raw-out "$OUT_NPZ"
echo ">>> DONE — $OUT_JSON + $OUT_NPZ"
