#!/bin/bash
# Serve the LoRA-off-GSE π0.5 policy (job 383) over the openpi WebSocket so the
# FARM dashboard can drive the arm with it. Thin wrapper around serve_pi05.sbatch
# — it only pins the config/checkpoint/prompt, then hands off to the shared serve
# logic (checkpoint resolution + NGC container + serve_policy.py on :8000).
#
# This is the LoRA fine-tuned ON TOP of the GSE multiobject checkpoint
# (GSEMergeWeightLoader merged the GSE adapters into the dense backbone, then a
# plain rank-16 LoRA trained on the 100 bottle->box episodes = multiobject[:100]).
#
#   ⚠ BOTTLE-ONLY. It trained on 100 bottle->box demos and nothing else — only the
#     "Picking up the bottle and placing it on the box" task works. The bear/duck/
#     hat dashboard presets will no-op or flail with this checkpoint.
#
#   ⚠ Checkpoint = step-2000 by default. The real-episode eval
#     (analysis/realdata_lora_gse/FINDINGS.md) found the LoRA over-specialized on
#     its 100 episodes: step-2000 generalizes best, held-out error PEAKS at 6000,
#     and 9999 is the most overfit. Override with STEP=4000|6000|8000|9999 if the
#     live bottle layout closely matches the demonstrated motions (tighter fit).
#
#   ⚠ LOCAL-ONLY checkpoint. Job 383's checkpoints were never pushed to HF, so
#     there is no download fallback — they must exist on the Weka FS at the path
#     below (they do, from training). If you wipe them, re-run job 383.
#
# Submit:    sbatch serve_pi05_lora_gse.sbatch                 # step-2000
#            STEP=9999 sbatch serve_pi05_lora_gse.sbatch       # a different step
# Logs:      tail -f /home/nhweiss/farm-train/serve-<jobid>.out
# Port:      8000 — port-forward to the laptop per model/cluster/DEPLOYMENT.md §D.

#SBATCH --job-name=serve-lora-gse
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/nhweiss/farm-train/serve-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
STEP="${STEP:-2000}"

export SERVE_CONFIG="pi05_farm_bottle_lora_gse"
export SERVE_CKPT="$WORK/openpi/checkpoints/pi05_farm_bottle_lora_gse/pi05_farm_bottle_lora_gse_383/${STEP}"
export SERVE_PROMPT="Picking up the bottle and placing it on the box"

# ── Pre-flight: fail loud BEFORE burning the container build / GPU minutes. ──
CFG_PY="$WORK/openpi/src/openpi/training/config.py"
if ! grep -q "pi05_farm_bottle_lora_gse" "$CFG_PY" 2>/dev/null; then
  echo "FATAL: config 'pi05_farm_bottle_lora_gse' is not registered in $CFG_PY." >&2
  echo "       Re-apply the patches (the serve instantiates this TrainConfig, which" >&2
  echo "       references GSEMergeWeightLoader, so config.py must import cleanly):" >&2
  echo "         python model/cluster/patch_openpi_config.py" >&2
  echo "         python model/cluster/patch_openpi_gse.py" >&2
  echo "         python model/cluster/patch_openpi_gse_merge.py" >&2
  echo "         python model/cluster/patch_openpi_config_pi05_lora_bottle100_gse.py" >&2
  exit 1
fi
if [[ ! -d "$SERVE_CKPT/params" ]]; then
  echo "FATAL: no checkpoint params at $SERVE_CKPT/params" >&2
  echo "       Available steps:" >&2
  ls -1d "$WORK"/openpi/checkpoints/pi05_farm_bottle_lora_gse/pi05_farm_bottle_lora_gse_383/*/ 2>/dev/null >&2 || true
  exit 1
fi

echo ">>> LoRA-off-GSE serve · config=$SERVE_CONFIG · step=$STEP"
echo ">>> checkpoint: $SERVE_CKPT"

# Hand off to the shared serve logic. The inner script's #SBATCH lines are inert
# comments here (we are already inside this job's allocation); it resolves the
# checkpoint from SERVE_CKPT, then runs serve_policy.py on :8000 in the NGC image.
exec bash "$WORK/serve_pi05.sbatch"
