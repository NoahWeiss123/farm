#!/bin/bash
# Benchmark eval (inside the NGC container): FFT vs LoRA vs multiobject-GSE on
# eval_bench (15 eps, 5 tasks). For each model: resolve its checkpoint (local
# highest-step, else HF download), run clean + robust eval → JSONs for make_report.
set -uo pipefail
WORK="$(cd "$(dirname "$0")" && pwd)"
export HOME="$(dirname "$WORK")"
source "$WORK/.hf_env"
export HF_HUB_ENABLE_HF_TRANSFER=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Eval is inference (train=False) → augmentation/prompt-paraphrase OFF.
unset FARM_AUG_LEVEL FARM_PROMPT_AUG

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(full lora multiobject)
declare -A CONF=( [full]=pi05_farm_uf850 [lora]=pi05_farm_bottle_lora [multiobject]=pi05_farm_multiobject_gse )
declare -A REPO=( [full]=NoahWeiss/farm_uf850_pi05 [lora]="" [multiobject]=NoahWeiss/farm_uf850_multiobject_gse_robust )
declare -A DLSTEP=( [full]=19999 [lora]=10000 [multiobject]=6000 )
# Local checkpoint parent dirs (we take the highest-numbered step subdir).
declare -A PARENT=( [full]="checkpoints/pi05_farm_uf850"
                    [lora]="checkpoints/pi05_farm_bottle_lora/pi05_farm_bottle_lora_188"
                    [multiobject]="checkpoints/pi05_farm_multiobject_gse/farm_gse_multiobject_robust_190" )

highest_step() {  # echo the path of the highest-numbered step dir under $1 (recursively, depth<=2)
  find "$1" -maxdepth 2 -type d -regex '.*/[0-9]+$' 2>/dev/null \
    | awk -F/ '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2-
}

set -e
echo ">>> ffmpeg…"; apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv; cd "$WORK/openpi"; uv sync --frozen
set +e

for m in "${MODELS[@]}"; do
  cfg="${CONF[$m]:-}"; [ -z "$cfg" ] && { echo "!! unknown $m"; continue; }
  ckpt="$(highest_step "$WORK/openpi/${PARENT[$m]}")"
  if [ -z "$ckpt" ] || [ ! -d "$ckpt/params" ]; then
    repo="${REPO[$m]}"; step="${DLSTEP[$m]}"
    if [ -n "$repo" ]; then
      echo ">>> [$m] no local ckpt; downloading step-$step from $repo…"
      dl="$WORK/checkpoints/evalbench_${m}"
      hf download "$repo" --include "step-${step}/*" --local-dir "$dl" >/dev/null 2>&1
      ckpt="$dl/step-${step}"
    fi
  fi
  if [ ! -d "$ckpt/params" ]; then echo "!! [$m] no checkpoint resolved ($ckpt) — skipping"; continue; fi
  echo "================ EVAL $m  cfg=$cfg  ckpt=$ckpt ================"
  uv run python "$WORK/eval_offline.py" --config "$cfg" --checkpoint-dir "$ckpt" \
      --episodes-dir "$WORK/eval_bench" --model "$m" --out "$WORK/eval-clean-${m}.json" \
      --n-episodes 15 --samples-per-episode 16
  uv run python "$WORK/eval_robust.py" --config "$cfg" --checkpoint-dir "$ckpt" \
      --episodes-dir "$WORK/eval_bench" --model "$m" --out "$WORK/eval-robust-${m}.json" \
      --n-episodes 15 --samples-per-episode 8
  echo "================ EVAL $m done ================"
done
echo ">>> ALL BENCH EVALS DONE"
