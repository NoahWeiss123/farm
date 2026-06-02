#!/bin/bash
# FFT checkpoint GENERALIZATION sweep (1 GPU). Tests EVERY saved FFT checkpoint on
# real episodes and reports which performs best, two complementary ways:
#   (A) TRAINING-DATA episodes — eval_train_endhorizon over a fixed sample of the
#       multi-object set (real demonstrated frames the FFT trained on). Same seed
#       across checkpoints → identical episodes/frames → a clean per-checkpoint
#       action-accuracy curve. (The user's ask: "test against episodes from the
#       training data and see which one performs best.")
#   (B) [done separately by eval_fft_bench.sbatch fftonly] HELD-OUT eval_bench
#       (separate recordings + an OOD task) clean + domain-shift — the true
#       generalization cross-check, so we don't just reward memorization.
#
# Because the FFT trained on ALL 424 multi-object episodes, (A) measures FIT and
# will tend to favour later steps; (B) measures GENERALIZATION and may peak
# earlier. select_fft_base.py combines them → the best-generalizing checkpoint
# (the LoRA base / lora_base), reported with the full curve.
#
#   sbatch eval_fft_ckptgen.sbatch ; then select_fft_base.py over the JSONs.
set -uo pipefail
WORK="$(cd "$(dirname "$0")" && pwd)"
export HOME="$(dirname "$WORK")"
source "$WORK/.hf_env"
export HF_HUB_ENABLE_HF_TRANSFER=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
unset FARM_AUG_LEVEL FARM_PROMPT_AUG FARM_EP_RANGE

REPO="NoahWeiss/farm_uf850_multiobject"
FFT_DIR="$WORK/openpi/checkpoints/pi05_farm_multiobject_fft/farm_fft_multiobject_robust_406"
OUT="$WORK/fft_ckptgen"; mkdir -p "$OUT"
SEED="${SEED:-0}"; NEPS="${NEPS:-48}"; SPE="${SPE:-8}"

set -e
echo ">>> ffmpeg…"; apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv; cd "$WORK/openpi"; uv sync --frozen
set +e

steps=$(ls "$FFT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
[ -z "$steps" ] && { echo "!! no FFT checkpoints in $FFT_DIR"; exit 1; }
echo ">>> sweeping checkpoints: $steps  (seed=$SEED, $NEPS eps × $SPE frames)"
for s in $steps; do
  ck="$FFT_DIR/$s"; [ -d "$ck/params" ] || { echo "skip $s (no params)"; continue; }
  echo "===== FFT step $s · training-data action accuracy ====="
  uv run python "$WORK/eval_train_endhorizon.py" --config pi05_farm_multiobject_fft --checkpoint-dir "$ck" \
    --repo-id "$REPO" --n-episodes "$NEPS" --samples-per-episode "$SPE" --horizon 10 \
    --roll-per-task 0 --seed "$SEED" --model "fft_$s" --split train \
    --out "$OUT/traindata-fft-$s.json" --raw-out "$OUT/traindata-fft-$s.npz"
done
echo ">>> FFT CKPTGEN SWEEP DONE → $OUT"
