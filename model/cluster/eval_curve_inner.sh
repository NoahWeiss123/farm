#!/bin/bash
# eval_curve_inner.sh — produce the error-vs-training-step curve for ONE model.
#
# For each checkpoint STEP it runs eval_train_endhorizon.py TWICE (same metric,
# same seed/frames) — once on the "train" head episodes and once on the
# "held_out" tail episodes — and writes one JSON per (split, step):
#     $RESULTS_DIR/eval-<split>-<MODEL_TAG>-step<STEP>.json
# The plotted scalar is end_of_horizon.overall_joint_mae_deg.
#
# Inputs (env):
#   MODEL_TAG      e2 | e8 | e32 | e64 | fft         (label in JSON + filename)
#   EVAL_CONFIG    pi05_farm_multiobject_gse_sweep | pi05_farm_multiobject_fft
#   FARM_GSE_NUM_SPECIALIZED   1|7|31|63 for GSE; MUST be unset/empty for fft
#   SOURCE_KIND    local | hf
#   EXP_DIR        (local) parent dir holding <STEP>/ checkpoint subdirs
#   HF_REPO        (hf)    e.g. NoahWeiss/farm_uf850_multiobject_fft_robust
#   RESULTS_DIR    where JSONs are written
#   STEPS          space-separated (default: 8000 16000 24000 32000 40000 48000 55999)
#   TRAIN_RANGES   default per-task heads  "0:239,299:339,349:374,384:414"
#   HELDOUT_RANGES default per-task tails  "239:299,339:349,374:384,414:424"
#   N_EPISODES     default 80   SAMPLES default 10   SEED default 0
set -uo pipefail

WORK="$HOME/farm-train"
REPO_ID="NoahWeiss/farm_uf850_multiobject"
: "${MODEL_TAG:?}"; : "${EVAL_CONFIG:?}"; : "${SOURCE_KIND:?}"; : "${RESULTS_DIR:?}"
STEPS="${STEPS:-8000 16000 24000 32000 40000 48000 55999}"
TRAIN_RANGES="${TRAIN_RANGES:-0:239,299:339,349:374,384:414}"
HELDOUT_RANGES="${HELDOUT_RANGES:-239:299,339:349,374:384,414:424}"
N_EPISODES="${N_EPISODES:-80}"; SAMPLES="${SAMPLES:-10}"; SEED="${SEED:-0}"
DL_ROOT="${DL_ROOT:-$WORK/eval_curve_dl/$MODEL_TAG}"
mkdir -p "$RESULTS_DIR"

echo ">>> eval_curve model=$MODEL_TAG config=$EVAL_CONFIG source=$SOURCE_KIND ns=${FARM_GSE_NUM_SPECIALIZED:-<unset>}"
echo ">>> steps: $STEPS"

run_one() {  # split ranges step ckpt
  local split="$1" ranges="$2" step="$3" ckpt="$4"
  local out="$RESULTS_DIR/eval-${split}-${MODEL_TAG}-step${step}.json"
  if [ -s "$out" ]; then echo "    [$split step $step] exists, skip"; return 0; fi
  echo "    [$split step $step] -> $(basename "$out")"
  uv run python "$WORK/eval_train_endhorizon.py" \
    --config "$EVAL_CONFIG" --checkpoint-dir "$ckpt" --repo-id "$REPO_ID" \
    --split "$split" --ep-range "$ranges" \
    --n-episodes "$N_EPISODES" --samples-per-episode "$SAMPLES" --horizon 10 \
    --roll-per-task 1 --roll-stride 4 --seed "$SEED" --model "$MODEL_TAG" \
    --out "$out" || { echo "    !! eval failed ($split step $step)"; return 1; }
}

for STEP in $STEPS; do
  # resume-friendly: skip the whole step (no download) if both splits are done
  if [ -s "$RESULTS_DIR/eval-train-${MODEL_TAG}-step${STEP}.json" ] && \
     [ -s "$RESULTS_DIR/eval-held_out-${MODEL_TAG}-step${STEP}.json" ]; then
    echo ">> step $STEP: both JSONs exist — skip"; continue
  fi
  # resolve checkpoint dir (must contain params/ + assets/)
  if [ "$SOURCE_KIND" = "local" ]; then
    : "${EXP_DIR:?}"; CKPT="$EXP_DIR/$STEP"
    if [ ! -d "$CKPT/params" ]; then echo ">> step $STEP: no local $CKPT/params — skip"; continue; fi
    CLEANUP=""
  else
    : "${HF_REPO:?}"; CKPT="$DL_ROOT/step-$STEP"
    # Treat the download as complete ONLY if orbax's params/_METADATA index is
    # present. Wipe before each attempt so a truncated/partial dir is never reused.
    if [ ! -f "$CKPT/params/_METADATA" ]; then
      echo ">> step $STEP: downloading step-$STEP from $HF_REPO …"
      ok=0
      for attempt in 1 2 3 4 5; do
        rm -rf "$CKPT"
        uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_REPO', allow_patterns=['step-$STEP/*'], local_dir='$DL_ROOT', max_workers=4)" >/dev/null 2>&1
        [ -f "$CKPT/params/_METADATA" ] && { ok=1; break; }
        echo ">> step $STEP: download attempt $attempt incomplete (no params/_METADATA); retry in 30s…"; sleep 30
      done
      [ "$ok" = 1 ] || { echo ">> step $STEP: download failed/incomplete after 5 retries — skip"; continue; }
    fi
    [ -d "$CKPT/params" ] || { echo ">> step $STEP: no params after download — skip"; continue; }
    CLEANUP="$CKPT"
  fi

  run_one train    "$TRAIN_RANGES"   "$STEP" "$CKPT"
  run_one held_out "$HELDOUT_RANGES" "$STEP" "$CKPT"

  # bound disk for HF source: drop the downloaded checkpoint once evaluated
  [ -n "${CLEANUP:-}" ] && rm -rf "$CLEANUP"
done

echo ">>> eval_curve done for $MODEL_TAG — JSONs in $RESULTS_DIR"
ls -1 "$RESULTS_DIR" | grep -- "-${MODEL_TAG}-" || true
