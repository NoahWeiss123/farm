#!/bin/bash
# SLURM batch file: serve the trained π0.5 policy over an openpi WebSocket.
#
# THIS IS THE STEP THAT ACTIVATES A GPU. Everything else (model staged,
# client ready, network path documented) is set up ahead of this — see
# tools/cluster/DEPLOYMENT.md. Submit this only when you're ready to run.
#
# Sized for 1× H100 — batch-1 inference of a ~3.3B π0.5 is trivial; the
# whole node isn't needed. 4h walltime (resubmit for longer sessions).
#
# Submit:    sbatch serve_pi05.sbatch
# Logs:      tail -f /home/nhweiss/farm-train/serve-<jobid>.out
# Port:      8000 (openpi WebSocket default) — port-forward per DEPLOYMENT.md

#SBATCH --job-name=serve-pi05
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/nhweiss/farm-train/serve-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"            # HF_TOKEN — only needed if the HF fallback download runs
CONFIG_NAME="pi05_farm_uf850"

# ── Resolve the checkpoint to serve (final model = step 19999) ──────────
# Preference order, so this works whether or not the training dir was kept:
#   1) the local training output (no copy/download — it's right there)
#   2) a previously-downloaded stable path
#   3) download step-19999 from HF (public repo; the verified source of truth)
LOCAL_CKPT=$(ls -d "$WORK"/openpi/checkpoints/pi05_farm_uf850/*/19999 2>/dev/null | head -1)
STABLE_CKPT="$WORK/checkpoints/pi05_step19999/step-19999"
if [[ -n "$LOCAL_CKPT" && -d "$LOCAL_CKPT/params" ]]; then
  CKPT_DIR="$LOCAL_CKPT"
elif [[ -d "$STABLE_CKPT/params" ]]; then
  CKPT_DIR="$STABLE_CKPT"
else
  echo ">>> no local checkpoint — downloading step-19999 from HF…"
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download NoahWeiss/farm_uf850_pi05 \
      --include 'step-19999/*' --local-dir "$WORK/checkpoints/pi05_step19999"
  CKPT_DIR="$STABLE_CKPT"
fi
echo ">>> serving checkpoint: $CKPT_DIR"

# ── Launch the policy server inside the NGC container ───────────────────
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

# Bind 0.0.0.0 so kubectl port-forward from outside the pod reaches it.
# --default_prompt is the fallback task if the client omits a prompt; the
# eval client always sends its own --task, which overrides this.
echo '>>> serve_policy.py on :8000 (config=${CONFIG_NAME}, ckpt=${CKPT_DIR})'
uv run python scripts/serve_policy.py \
    --port 8000 \
    --default_prompt 'Picking up the bottle and placing it on the box' \
    policy:checkpoint \
    --policy.config=${CONFIG_NAME} \
    --policy.dir='${CKPT_DIR}'
"
