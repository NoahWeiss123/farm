#!/bin/bash
# Offline action-accuracy eval of the GSE π0.5 checkpoint on training episodes.
#
# In-process (no serve_policy / websocket): loads the checkpoint and replays
# recorder-format episodes staged at ~/farm-train/eval_episodes, comparing
# predicted actions to the recorded ground truth. 1× H100, short.
#
# Stage first (from the laptop):
#   - model/cluster/eval_offline.py  -> ~/farm-train/eval_offline.py
#   - a handful of episode_* dirs    -> ~/farm-train/eval_episodes/
# Submit:  sbatch eval_pi05_gse.sbatch
# Logs:    tail -f ~/farm-train/eval-gse-<jobid>.out

#SBATCH --job-name=eval-pi05-gse
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/nhweiss/farm-train/eval-gse-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"            # HF_TOKEN — only if the HF fallback runs
CONFIG_NAME="pi05_farm_uf850_gse"
REPO_ID="NoahWeiss/farm_uf850_pi05_gse"
STEP=2999

# ── Resolve the checkpoint (local training output first; HF as fallback) ──
LOCAL_CKPT=$(ls -d "$WORK"/openpi/checkpoints/pi05_farm_uf850_gse/*/${STEP} 2>/dev/null | head -1)
STABLE_CKPT="$WORK/checkpoints/gse_step${STEP}/step-${STEP}"
if [[ -n "$LOCAL_CKPT" && -d "$LOCAL_CKPT/params" ]]; then
  CKPT_DIR="$LOCAL_CKPT"
elif [[ -d "$STABLE_CKPT/params" ]]; then
  CKPT_DIR="$STABLE_CKPT"
else
  echo ">>> no local checkpoint — downloading step-${STEP} from HF…"
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$REPO_ID" \
      --include "step-${STEP}/*" --local-dir "$WORK/checkpoints/gse_step${STEP}"
  CKPT_DIR="$STABLE_CKPT"
fi
echo ">>> evaluating checkpoint: $CKPT_DIR"

# ── Run the eval inside the openpi container (needs JAX + GPU) ────────────
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

uv run python \$HOME/farm-train/eval_offline.py \
    --config ${CONFIG_NAME} \
    --checkpoint-dir '${CKPT_DIR}' \
    --episodes-dir \$HOME/farm-train/eval_episodes \
    --n-episodes 6 --samples-per-episode 16 --horizon 10
"
echo ">>> eval done"
