#!/bin/bash
# In-container eval (1 GPU): base FFT-56k vs FFT-56k+LoRA on HELD-OUT episodes of
# each task — episodes OUTSIDE each LoRA's training slice but inside the multi-object
# set. Same seed for every (base, lora) pair → identical sampled frames → a clean
# paired comparison. Uses eval_train_endhorizon.py (real LeRobot frames, real
# ground-truth chunks). Held-out ranges (all GENUINE held-out with the n=30 set):
#   bottle: LoRA fftlora_bottle100 trained 0:100   → held-out 100:299 (199 eps) [PRIMARY]
#   bear:   LoRA fftlora_bear30   trained 299:329 → held-out 329:349 (20 eps)
#   duck:   LoRA fftlora_duck30   trained 349:379 → held-out 379:384 (5 eps, small)
#   hat:    LoRA fftlora_hat30    trained 384:414 → held-out 414:424 (10 eps)
#
# JSON+NPZ per run → ~/farm-train/fftlora_analysis/. FAILS LOUDLY if the FFT base
# checkpoint is missing (so a missing base can't silently yield a LoRA-only result).
set -uo pipefail
WORK="$(cd "$(dirname "$0")" && pwd)"
export HOME="$(dirname "$WORK")"
source "$WORK/.hf_env"
export HF_HUB_ENABLE_HF_TRANSFER=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
unset FARM_AUG_LEVEL FARM_PROMPT_AUG FARM_EP_RANGE   # eval = inference; load full set, window via --ep-range

REPO="NoahWeiss/farm_uf850_multiobject"
OUT="$WORK/fftlora_analysis"; mkdir -p "$OUT"
SEED="${SEED:-0}"; SPE="${SPE:-10}"
# Base = the SELECTED best FFT checkpoint (lora_base symlink) — the SAME checkpoint
# the LoRAs were trained on top of, so base-vs-(base+LoRA) is apples-to-apples.
FFT_CKPT="$WORK/openpi/checkpoints/pi05_farm_multiobject_fft/farm_fft_multiobject_robust_406/lora_base"
[ -d "$FFT_CKPT/params" ] || { echo "FATAL: FFT base missing at $FFT_CKPT/params — set the lora_base symlink (post-selection) first. Aborting."; exit 1; }

highest_step() { find "$1" -maxdepth 2 -type d -regex '.*/[0-9]+$' 2>/dev/null \
  | awk -F/ '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2-; }

run() {  # label config ckpt eprange neps split
  local label="$1" cfg="$2" ck="$3" epr="$4" neps="$5" split="$6"
  [ -d "$ck/params" ] || { echo "!! [$label] no ckpt at $ck — skipping"; return 1; }
  echo "===== EVAL $label  cfg=$cfg  range=$epr  split=$split  ckpt=$ck ====="
  uv run python "$WORK/eval_train_endhorizon.py" --config "$cfg" --checkpoint-dir "$ck" \
    --repo-id "$REPO" --ep-range "$epr" --n-episodes "$neps" --samples-per-episode "$SPE" \
    --roll-per-task 0 --seed "$SEED" --model "$label" --split "$split" \
    --out "$OUT/cmp-$label.json" --raw-out "$OUT/cmp-$label.npz"
}

set -e
echo ">>> ffmpeg…"; apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv; cd "$WORK/openpi"; uv sync --frozen
set +e

# task | held-out range | LoRA exp-name | NEPS (bottle bigger — 199 held-out eps)
# label fields: split=held_out for all (n=30 set leaves every task a held-out tail).
for spec in \
  "bottle 100:299 fftlora_bottle100 60" \
  "bear 329:349 fftlora_bear30 20" \
  "duck 379:384 fftlora_duck30 5" \
  "hat 414:424 fftlora_hat30 10" ; do
  set -- $spec; task="$1"; epr="$2"; lexp="$3"; neps="$4"
  run "fftbase_${task}" pi05_farm_multiobject_fft "$FFT_CKPT" "$epr" "$neps" held_out \
    || { echo "FATAL: base eval failed for $task"; exit 1; }
  lck="$(highest_step "$WORK/openpi/checkpoints/pi05_fftlora/${lexp}")"
  [ -n "$lck" ] && run "fftlora_${task}" pi05_fftlora "$lck" "$epr" "$neps" held_out
done
echo ">>> FFTLORA COMPARE EVALS DONE → $OUT"
