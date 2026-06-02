#!/bin/bash
# Real-episode prediction eval for the bottle LoRAs. Arg: gse | base | fit15.
#  gse/base  : held-out probe multiobject[100:299] (same camera, unseen episodes)
#  fit15     : WITHIN-TRAINING — 15 episodes the LoRAs DID train on
#              (farm_bottle_lora = multiobject[0:100]); per-episode MAE for the
#              average + outlier analysis. Runs GSE-init LoRA, base-init LoRA, and
#              the GSE model on the SAME 15 episodes (seed-fixed) for comparison.
set -uo pipefail
WORK="$(cd "$(dirname "$0")" && pwd)"
export HOME="$(dirname "$WORK")"
[ -f "$WORK/.hf_env" ] && source "$WORK/.hf_env"
export HF_HUB_ENABLE_HF_TRANSFER=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
unset FARM_AUG_LEVEL FARM_PROMPT_AUG

MODEL="${1:?usage: eval_lora_runner.sh gse|base|fit15}"
LG="openpi/checkpoints/pi05_farm_bottle_lora_gse/pi05_farm_bottle_lora_gse_383"
LB="openpi/checkpoints/pi05_farm_bottle_lora/pi05_farm_bottle_lora_188"
GM="openpi/checkpoints/pi05_farm_multiobject_gse/farm_gse_multiobject_robust_190"

if [ "$MODEL" = "fit15" ]; then
  REPO="NoahWeiss/farm_bottle_lora"; EPR=""          # all 100 training eps
  # TAG CONFIG ROOT STEP N S ROLL  (N=15 eps, 30 samples each, +2 rollouts)
  SPECS="fit15_gse pi05_farm_bottle_lora_gse $LG 9999 15 30 2
fit15_base pi05_farm_bottle_lora $LB 9999 15 30 2
fit15_gsemodel pi05_farm_multiobject_gse $GM 5999 15 30 2"
elif [ "$MODEL" = "gse" ]; then
  REPO="NoahWeiss/farm_uf850_multiobject"; EPR="100:299"
  SPECS="gse_heldout pi05_farm_bottle_lora_gse $LG 9999 64 12 1
gse_heldout_2000 pi05_farm_bottle_lora_gse $LG 2000 40 8 0
gse_heldout_4000 pi05_farm_bottle_lora_gse $LG 4000 40 8 0
gse_heldout_6000 pi05_farm_bottle_lora_gse $LG 6000 40 8 0
gse_heldout_8000 pi05_farm_bottle_lora_gse $LG 8000 40 8 0
gsemodel_heldout_fit pi05_farm_multiobject_gse $GM 5999 64 12 1"
else
  REPO="NoahWeiss/farm_uf850_multiobject"; EPR="100:299"
  SPECS="base_heldout pi05_farm_bottle_lora $LB 9999 64 12 1
base_heldout_2000 pi05_farm_bottle_lora $LB 2000 40 8 0
base_heldout_4000 pi05_farm_bottle_lora $LB 4000 40 8 0
base_heldout_6000 pi05_farm_bottle_lora $LB 6000 40 8 0
base_heldout_8000 pi05_farm_bottle_lora $LB 8000 40 8 0"
fi

set -e
echo ">>> ffmpeg…"; apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv; cd "$WORK/openpi"; uv sync --frozen
set +e

echo "================ LoRA eval · MODEL=$MODEL · ${REPO}${EPR:+[$EPR]} ================"
printf '%s\n' "$SPECS" | while read -r TAG CFG ROOT STEP N S ROLL; do
  [ -z "${TAG:-}" ] && continue
  CKPT="$WORK/$ROOT/$STEP"
  if [ ! -d "$CKPT/params" ]; then echo "!! missing $CKPT/params — skip $TAG"; continue; fi
  EPARG=""; [ -n "$EPR" ] && EPARG="--ep-range $EPR"
  RAW=""; [ "${ROLL:-0}" -gt 0 ] && RAW="--raw-out $WORK/eval-LORA-$TAG-raw.npz"
  echo ">>>> [$TAG] config=$CFG step=$STEP N=$N S=$S roll=$ROLL ${EPR:+eprange=$EPR}"
  uv run python "$WORK/eval_train_endhorizon.py" \
     --config "$CFG" --checkpoint-dir "$CKPT" --repo-id "$REPO" $EPARG \
     --n-episodes "$N" --samples-per-episode "$S" --horizon 10 \
     --roll-per-task "${ROLL:-0}" --roll-stride 3 --model "$TAG" \
     --out "$WORK/eval-LORA-$TAG.json" $RAW \
     || echo "!! eval $TAG FAILED (continuing)"
done
echo ">>> ALL EVALS DONE for MODEL=$MODEL"
