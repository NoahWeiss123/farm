#!/bin/bash
# SLURM batch file: serve the trained π0.5 policy over an openpi WebSocket.
#
# THIS IS THE STEP THAT ACTIVATES A GPU. Everything else (model staged,
# client ready, network path documented) is set up ahead of this — see
# model/cluster/DEPLOYMENT.md. Submit this only when you're ready to run.
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
# Parameterized by env. DEFAULT = the overnight domain-robust FLAGSHIP
# (GSE + heavy domain-randomization + prompt-aug). It's the best bet when the
# deploy scene differs from the training room (the actual live-failure case):
# it barely degrades under a room/lighting change where the full FT degrades ~2×
# (see analysis/overnight/farm_pi05_domain_robustness.pdf).
# To serve the ORIGINAL full fine-tune instead (tighter fit IN the training room):
#   SERVE_CONFIG=pi05_farm_uf850 SERVE_REPO=NoahWeiss/farm_uf850_pi05 \
#   SERVE_STEP=19999 SERVE_EXP='*' sbatch serve_pi05.sbatch
#   SERVE_CONFIG  registered openpi config   SERVE_STEP   checkpoint step
#   SERVE_REPO    HF repo for the fallback   SERVE_PROMPT default prompt
CONFIG_NAME="${SERVE_CONFIG:-pi05_farm_uf850_gse}"
REPO_ID="${SERVE_REPO:-NoahWeiss/farm_uf850_pi05_gse_robust}"
STEP="${SERVE_STEP:-2999}"
PROMPT="${SERVE_PROMPT:-Picking up the bottle and placing it on the box}"

# ── Resolve the checkpoint to serve (local training output → stable → HF) ──
# Several exp dirs can share one config+step (the 3 overnight ablations all live
# under pi05_farm_uf850_gse), so a bare */STEP glob is ambiguous. SERVE_EXP picks
# the exp-dir (default *robust* = the flagship); SERVE_CKPT overrides outright.
# For the full FT, pass SERVE_EXP='*'.
EXP_GLOB="${SERVE_EXP:-*robust*}"
if [[ -n "${SERVE_CKPT:-}" ]]; then
  LOCAL_CKPT="$SERVE_CKPT"
else
  LOCAL_CKPT=$(ls -dt "$WORK"/openpi/checkpoints/${CONFIG_NAME}/${EXP_GLOB}/${STEP} 2>/dev/null | head -1)
fi
STABLE_CKPT="$WORK/checkpoints/${CONFIG_NAME}_step${STEP}/step-${STEP}"
if [[ -n "$LOCAL_CKPT" && -d "$LOCAL_CKPT/params" ]]; then
  CKPT_DIR="$LOCAL_CKPT"
elif [[ -d "$STABLE_CKPT/params" ]]; then
  CKPT_DIR="$STABLE_CKPT"
else
  echo ">>> no local checkpoint — downloading step-${STEP} from ${REPO_ID}…"
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$REPO_ID" \
      --include "step-${STEP}/*" --local-dir "$WORK/checkpoints/${CONFIG_NAME}_step${STEP}"
  CKPT_DIR="$STABLE_CKPT"
fi
echo ">>> serving checkpoint: $CKPT_DIR  (config=$CONFIG_NAME)"

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
    --default_prompt '${PROMPT}' \
    policy:checkpoint \
    --policy.config=${CONFIG_NAME} \
    --policy.dir='${CKPT_DIR}'
"
