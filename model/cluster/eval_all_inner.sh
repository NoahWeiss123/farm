#!/bin/bash
# Runs INSIDE the NGC container for eval_all.sbatch. Args = model labels to eval.
# For each: resolve a checkpoint (local-first, else HF download), then run the
# clean-fit eval and the domain-shift robustness eval.

set -uo pipefail
WORK="$(cd "$(dirname "$0")" && pwd)"
export HOME="$(dirname "$WORK")"
source "$WORK/.hf_env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(full gse gse_robust gse_aug gse_prompt)

# label → config / repo / step / local-glob
declare -A CONF=(  [full]=pi05_farm_uf850 [gse]=pi05_farm_uf850_gse
                   [gse_robust]=pi05_farm_uf850_gse [gse_aug]=pi05_farm_uf850_gse
                   [gse_prompt]=pi05_farm_uf850_gse )
declare -A REPO=(  [full]=NoahWeiss/farm_uf850_pi05 [gse]=NoahWeiss/farm_uf850_pi05_gse
                   [gse_robust]=NoahWeiss/farm_uf850_pi05_gse_robust
                   [gse_aug]=NoahWeiss/farm_uf850_pi05_gse_aug
                   [gse_prompt]=NoahWeiss/farm_uf850_pi05_gse_prompt )
declare -A STEP=(  [full]=19999 [gse]=2999 [gse_robust]=2999 [gse_aug]=2999 [gse_prompt]=2999 )
declare -A GLOB=(  [full]="checkpoints/pi05_farm_uf850/*/19999"
                   [gse]="checkpoints/pi05_farm_uf850_gse/farm_uf850_pi05_gse_*/2999"
                   [gse_robust]="checkpoints/pi05_farm_uf850_gse/farm_gse_robust_*/2999"
                   [gse_aug]="checkpoints/pi05_farm_uf850_gse/farm_gse_augonly_*/2999"
                   [gse_prompt]="checkpoints/pi05_farm_uf850_gse/farm_gse_promptonly_*/2999" )

set -e
echo ">>> ffmpeg…"; apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd "$WORK/openpi"
uv sync --frozen
set +e

for m in "${MODELS[@]}"; do
  cfg="${CONF[$m]:-}"; repo="${REPO[$m]:-}"; step="${STEP[$m]:-}"
  if [ -z "$cfg" ]; then echo "!! unknown model label $m — skipping"; continue; fi
  ckpt="$(ls -d $WORK/openpi/${GLOB[$m]} 2>/dev/null | sort | tail -1)"
  if [ -z "$ckpt" ] || [ ! -d "$ckpt/params" ]; then
    echo ">>> [$m] no local ckpt; downloading step-$step from $repo…"
    dl="$WORK/checkpoints/eval_${m}"
    hf download "$repo" --include "step-${step}/*" --local-dir "$dl" >/dev/null 2>&1
    ckpt="$dl/step-${step}"
  fi
  if [ ! -d "$ckpt/params" ]; then echo "!! [$m] could not resolve a checkpoint — skipping"; continue; fi
  echo "================ EVAL $m  (config=$cfg)  ckpt=$ckpt ================"
  uv run python "$WORK/eval_offline.py" --config "$cfg" --checkpoint-dir "$ckpt" \
      --episodes-dir "$WORK/eval_episodes" --model "$m" \
      --out "$WORK/eval-clean-${m}.json" --n-episodes 6 --samples-per-episode 16
  uv run python "$WORK/eval_robust.py" --config "$cfg" --checkpoint-dir "$ckpt" \
      --episodes-dir "$WORK/eval_episodes" --model "$m" \
      --out "$WORK/eval-robust-${m}.json" --n-episodes 6 --samples-per-episode 8
  echo "================ EVAL $m done ================"
done
echo ">>> ALL EVALS DONE"
