#!/bin/bash
# Runs INSIDE the NGC container for train_overnight.sbatch. Standalone file so
# there is zero nested-quote escaping. Builds deps once, then trains three GSE
# augmentation variants sequentially, each streaming checkpoints to its own HF
# repo via a background pusher. A failed run is logged and the next still runs.

set -uo pipefail

# Locate self → derive workspace + HOME (container starts with HOME=/root).
WORK="$(cd "$(dirname "$0")" && pwd)"      # /home/<user>/farm-train
export HOME="$(dirname "$WORK")"           # /home/<user>
source "$WORK/.hf_env"                      # HF_TOKEN (+ HF_HUB_ENABLE_HF_TRANSFER)
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
JOB="${SLURM_JOB_ID:-manual}"
CONFIG="pi05_farm_uf850_gse"

# ── one-time setup (strict: abort the job if any of this fails) ──────────────
set -e
echo ">>> installing ffmpeg…"
apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd "$WORK/openpi"
uv sync --frozen
echo ">>> jax devices:"
uv run python -c "import jax; ds=jax.devices(); print(ds); assert len(ds) >= 4, f'want 4 GPUs, got {len(ds)}'"
echo ">>> norm stats (cached no-op if present)…"
uv run scripts/compute_norm_stats.py --config-name="$CONFIG"
set +e   # ── from here, a per-run failure must NOT kill the remaining runs ──

# run_one <label> <FARM_AUG_LEVEL> <FARM_PROMPT_AUG> <hf_repo> <steps>
run_one() {
  local label="$1" aug="$2" paug="$3" repo="$4" steps="$5"
  local exp="farm_gse_${label}_${JOB}"
  local ckpt="$WORK/openpi/checkpoints/${CONFIG}/${exp}"
  echo
  echo "################################################################"
  echo "## RUN ${label}: aug=${aug} prompt=${paug} steps=${steps}"
  echo "##   exp=${exp}  repo=${repo}"
  echo "################################################################"
  mkdir -p "$ckpt"
  # Background pusher → its own HF repo (keep-period matches config save_interval).
  uv run python "$WORK/push_checkpoints.py" \
      --checkpoint-dir "$ckpt" --repo-id "$repo" \
      --keep-period 1000 --poll-interval 60 \
      > "$WORK/push-${label}-${JOB}.out" 2>&1 &
  local pp=$!
  FARM_AUG_LEVEL="$aug" FARM_PROMPT_AUG="$paug" \
      uv run python scripts/train.py "$CONFIG" --exp-name="$exp" \
          --overwrite --no-wandb-enabled --num-train-steps="$steps"
  local rc=$?
  sleep 2; kill -TERM "$pp" 2>/dev/null; wait "$pp" 2>/dev/null
  echo "## RUN ${label} finished rc=${rc}"
}

# Flagship FIRST so the best model is available earliest. 2×2 cells:
#   (heavy,on)=robust  (heavy,off)=augonly  (default,on)=promptonly
#   (default,off)=vanilla GSE step-2999 — already on HF, no retrain.
run_one robust     heavy   1 NoahWeiss/farm_uf850_pi05_gse_robust 3000
run_one augonly    heavy   0 NoahWeiss/farm_uf850_pi05_gse_aug    3000
run_one promptonly default 1 NoahWeiss/farm_uf850_pi05_gse_prompt 3000

echo
echo ">>> ALL RUNS DONE (job ${JOB})"
