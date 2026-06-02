#!/bin/bash
# Sweep one model's checkpoints on TRAINING episodes (open-loop / teacher-forced
# action-prediction) via eval_train_endhorizon.py. Fixed seed + fixed sampling
# so EVERY model and checkpoint is scored on the IDENTICAL frames.
#
# Env in:
#   EVAL_CONFIG   registered openpi config (pi05_farm_multiobject_gse_sweep | pi05_farm_multiobject_fft)
#   EVAL_TAG      label for output files (e2,e4,e8,e16,e32,e80,fft)
#   EXP_DIR       checkpoint parent dir (contains integer step subdirs)
#   REPO_ID       training dataset (default NoahWeiss/farm_uf850_multiobject)
#   RESULTS_DIR   where JSON/NPZ land
#   FARM_GSE_NUM_SPECIALIZED   expert count for GSE models (so arch matches the ckpt); unset for FFT
#   SEED NEPS SPE H ROLLPT ROLLSTR   eval knobs (defaults below) — MUST be identical across all models
set -euo pipefail

: "${EVAL_CONFIG:?}"; : "${EVAL_TAG:?}"; : "${EXP_DIR:?}"
REPO_ID="${REPO_ID:-NoahWeiss/farm_uf850_multiobject}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/farm-train/eval_gse_sweep/$EVAL_TAG}"
SEED="${SEED:-0}"; NEPS="${NEPS:-80}"; SPE="${SPE:-10}"; H="${H:-10}"
ROLLPT="${ROLLPT:-1}"; ROLLSTR="${ROLLSTR:-4}"

mkdir -p "$RESULTS_DIR"
cd "$HOME/farm-train/openpi"

STEPS=$(ls -1 "$EXP_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
if [ -z "$STEPS" ]; then echo "!!! no integer step dirs under $EXP_DIR"; exit 2; fi
echo ">>> [$EVAL_TAG] config=$EVAL_CONFIG experts=${FARM_GSE_NUM_SPECIALIZED:-n/a}"
echo ">>> [$EVAL_TAG] seed=$SEED n_eps=$NEPS spe=$SPE H=$H  steps: $(echo $STEPS | tr '\n' ' ')"

# Assert the resolved expert count matches the env BEFORE scoring anything. At
# eval, create_trained_policy builds the GSE graph purely from
# FARM_GSE_NUM_SPECIALIZED (the SVD loader's shape-derivation runs only at train
# init) — all 6 GSE models share one config name, so a wrong/absent count would
# silently score a mismatched architecture. Fail loud first.
if [ "$EVAL_CONFIG" = "pi05_farm_multiobject_gse_sweep" ]; then
  : "${FARM_GSE_NUM_SPECIALIZED:?GSE eval requires FARM_GSE_NUM_SPECIALIZED}"
  uv run python -c "
import os
from openpi.models import gemma
ns = gemma.get_config('gemma_2b_gse').lora_configs['attn'].num_specialized
env = int(os.environ['FARM_GSE_NUM_SPECIALIZED'])
assert ns == env, f'EVAL WIRING MISMATCH: model num_specialized={ns} but env={env}'
print(f'EVAL_WIRING_OK num_specialized={ns} total_experts={ns+1}')
"
fi

for s in $STEPS; do
  CKPT="$EXP_DIR/$s"
  if [ ! -d "$CKPT/params" ]; then echo "  [$EVAL_TAG] skip $s (no params/)"; continue; fi
  OUT="$RESULTS_DIR/eval-train-${EVAL_TAG}-step${s}.json"
  RAW="$RESULTS_DIR/eval-train-${EVAL_TAG}-step${s}-raw.npz"
  if [ -f "$OUT" ]; then echo "  [$EVAL_TAG] step $s already done"; continue; fi
  echo ">>> [$EVAL_TAG] step $s → $(basename "$OUT")"
  uv run python "$HOME/farm-train/eval_train_endhorizon.py" \
    --config "$EVAL_CONFIG" --checkpoint-dir "$CKPT" \
    --repo-id "$REPO_ID" --split train \
    --n-episodes "$NEPS" --samples-per-episode "$SPE" --horizon "$H" \
    --roll-per-task "$ROLLPT" --roll-stride "$ROLLSTR" --seed "$SEED" \
    --model "$EVAL_TAG" --out "$OUT" --raw-out "$RAW"
done
echo ">>> [$EVAL_TAG] sweep complete → $RESULTS_DIR"
