#!/bin/bash
# In-container eval for the FFT-multiobject analysis (1 GPU, inference only).
#
#   (1) FFT CHECKPOINT SWEEP — eval every saved step of the full-FT run
#       (clean + domain-shift robust) on eval_bench → a training/robustness curve
#       and the basis for checkpoint selection.
#   (2) BASELINES — eval the 3 comparison techniques on the SAME eval_bench, in
#       the SAME openpi build, for a self-consistent 4-way head-to-head:
#         fftmulti  : THIS run (full FT, 424 eps/4 tasks, heavy aug + prompt aug)
#         multiobject: GSE-robust  (SAME data + aug; the controlled FT-method foil)
#         lora       : LoRA  (100-ep bottle)
#         full       : 2-task full FT (200-ep bottle; the over-memorised baseline)
#
# Eval is inference (train=False) → image aug + prompt paraphrase are OFF (the
# perturbations come from eval_robust.py, applied identically to every model).
# JSONs land in $WORK/fft_analysis/ for make_fft_report.py.
#
#   sbatch eval_fft_bench.sbatch                  # FFT sweep + all baselines
#   sbatch eval_fft_bench.sbatch fftonly         # FFT sweep only (faster)
#   tail -f eval-fft-bench-<jobid>.out
set -uo pipefail
WORK="$(cd "$(dirname "$0")" && pwd)"
export HOME="$(dirname "$WORK")"
source "$WORK/.hf_env"
export HF_HUB_ENABLE_HF_TRANSFER=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
unset FARM_AUG_LEVEL FARM_PROMPT_AUG   # inference: no train-time aug

MODE="${1:-all}"                       # "all" | "fftonly"
FFT_EXP="${FFT_EXP:-farm_fft_multiobject_robust_406}"
FFT_CFG="pi05_farm_multiobject_fft"
FFT_DIR="$WORK/openpi/checkpoints/$FFT_CFG/$FFT_EXP"
EVAL="$WORK/eval_bench"
OUT="$WORK/fft_analysis"; mkdir -p "$OUT"
NEPS=15

highest_step() { find "$1" -maxdepth 2 -type d -regex '.*/[0-9]+$' 2>/dev/null \
  | awk -F/ '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2-; }

run_clean()  { uv run python "$WORK/eval_offline.py" --config "$1" --checkpoint-dir "$2" \
                 --episodes-dir "$EVAL" --model "$3" --out "$OUT/eval-clean-$3.json" \
                 --n-episodes "$NEPS" --samples-per-episode 16; }
run_robust() { uv run python "$WORK/eval_robust.py" --config "$1" --checkpoint-dir "$2" \
                 --episodes-dir "$EVAL" --model "$3" --out "$OUT/eval-robust-$3.json" \
                 --n-episodes "$NEPS" --samples-per-episode 8; }

set -e
echo ">>> ffmpeg…"; apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv; cd "$WORK/openpi"; uv sync --frozen
[ -d "$EVAL" ] || { echo "!! eval_bench dir $EVAL missing"; exit 1; }
set +e

# ── (1) FFT checkpoint sweep ────────────────────────────────────────────────
echo "########## FFT CHECKPOINT SWEEP ($FFT_DIR) ##########"
steps=$(ls "$FFT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
[ -z "$steps" ] && echo "!! no FFT checkpoints under $FFT_DIR yet"
for s in $steps; do
  ck="$FFT_DIR/$s"; [ -d "$ck/params" ] || { echo "skip step $s (no params/)"; continue; }
  echo "===== FFT step $s · clean ====="; run_clean  "$FFT_CFG" "$ck" "fft_$s"
  echo "===== FFT step $s · robust ====="; run_robust "$FFT_CFG" "$ck" "fft_$s"
done
# Symlink the highest step as the canonical "fftmulti" for the head-to-head report.
best="$(echo "$steps" | tail -1)"
if [ -n "$best" ] && [ -f "$OUT/eval-clean-fft_$best.json" ]; then
  cp "$OUT/eval-clean-fft_$best.json"  "$OUT/eval-clean-fftmulti.json"
  cp "$OUT/eval-robust-fft_$best.json" "$OUT/eval-robust-fftmulti.json"
  echo ">>> fftmulti = highest step $best (override later via checkpoint selection)"
fi

[ "$MODE" = "fftonly" ] && { echo ">>> fftonly mode — skipping baselines"; echo ">>> FFT SWEEP DONE"; exit 0; }

# ── (2) baselines on the same eval_bench ────────────────────────────────────
echo "########## BASELINES ##########"
declare -A CFG=(  [multiobject]=pi05_farm_multiobject_gse [lora]=pi05_farm_bottle_lora [full]=pi05_farm_uf850 )
declare -A PARENT=( [multiobject]="checkpoints/pi05_farm_multiobject_gse/farm_gse_multiobject_robust_190"
                    [lora]="checkpoints/pi05_farm_bottle_lora/pi05_farm_bottle_lora_188"
                    [full]="checkpoints/pi05_farm_uf850" )
declare -A REPO=( [multiobject]=NoahWeiss/farm_uf850_multiobject_gse_robust [lora]="" [full]=NoahWeiss/farm_uf850_pi05 )
declare -A DLSTEP=( [multiobject]=5999 [lora]=10000 [full]=19999 )

for m in multiobject lora full; do
  ck="$(highest_step "$WORK/openpi/${PARENT[$m]}")"
  if [ -z "$ck" ] || [ ! -d "$ck/params" ]; then
    repo="${REPO[$m]}"; step="${DLSTEP[$m]}"
    if [ -n "$repo" ]; then
      echo ">>> [$m] no local ckpt; downloading step-$step from $repo…"
      dl="$WORK/checkpoints/evalfft_${m}"; hf download "$repo" --include "step-${step}/*" --local-dir "$dl" >/dev/null 2>&1
      ck="$dl/step-${step}"
    fi
  fi
  [ -d "$ck/params" ] || { echo "!! [$m] no checkpoint resolved ($ck) — skipping"; continue; }
  echo "===== baseline $m  cfg=${CFG[$m]}  ckpt=$ck ====="
  run_clean  "${CFG[$m]}" "$ck" "$m"
  run_robust "${CFG[$m]}" "$ck" "$m"
done
echo ">>> ALL FFT-BENCH EVALS DONE → $OUT"
