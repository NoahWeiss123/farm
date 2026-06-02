#!/bin/bash
# Serve the FFT multiobject full-finetune (training job 406) so the dashboard can
# drive the arm with it. This is the FULL fine-tune on all four objects
# (bottle / bear / duck / hat) — every dashboard task preset works, unlike the
# bottle-only LoRA-GSE serve.
#
# Thin wrapper over serve_pi05.sbatch: it pins the config and resolves a
# checkpoint, then hands off to the shared serve logic (NGC container +
# serve_policy.py on :8000).
#
#   STEP=latest (default) → newest checkpoint dir that has a complete params/
#     (skips a half-written one). As training keeps saving (48k, 55999) a fresh
#     serve picks up the newer checkpoint automatically.
#   STEP=<n>              → a specific step (8000/16000/24000/32000/40000/…).
#
# Serving a checkpoint from a STILL-RUNNING training job is fine: orbax step dirs
# are immutable once written, and we only pick one whose params/ already exists
# and that training has moved past.
#
# Submit:  sbatch serve_fft_multiobject.sbatch
#          STEP=32000 sbatch serve_fft_multiobject.sbatch
# Logs:    tail -f /home/nhweiss/farm-train/serve-<jobid>.out

#SBATCH --job-name=serve-fft-multiobj
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/home/nhweiss/farm-train/serve-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
CONFIG="pi05_farm_multiobject_fft"
EXP="$WORK/openpi/checkpoints/$CONFIG/farm_fft_multiobject_robust_406"
STEP="${STEP:-latest}"

# ── Resolve the checkpoint ──────────────────────────────────────────────────
if [[ "$STEP" == "latest" || -z "$STEP" ]]; then
  CKPT=""
  for d in $(ls -dt "$EXP"/*/ 2>/dev/null); do
    if [[ -d "${d}params" ]]; then CKPT="${d%/}"; break; fi   # newest complete
  done
else
  CKPT="$EXP/$STEP"
fi

# ── Pre-flight (fail before burning container build / GPU minutes) ──────────
if [[ -z "$CKPT" || ! -d "$CKPT/params" ]]; then
  echo "FATAL: no complete FFT checkpoint under $EXP (STEP=$STEP)" >&2
  echo "       available:" >&2
  ls -dt "$EXP"/*/ 2>/dev/null >&2 || true
  exit 1
fi
CFG_PY="$WORK/openpi/src/openpi/training/config.py"
if ! grep -q "$CONFIG" "$CFG_PY" 2>/dev/null; then
  echo "FATAL: config '$CONFIG' not registered in $CFG_PY" >&2
  exit 1
fi

export SERVE_CONFIG="$CONFIG"
export SERVE_CKPT="$CKPT"
export SERVE_PROMPT="Picking up the bottle and placing it on the box"

echo ">>> FFT multiobject serve · config=$CONFIG · STEP=$STEP"
echo ">>> checkpoint: $CKPT"

# Hand off to the shared serve logic (its #SBATCH lines are inert here — we are
# already inside this job's allocation).
exec bash "$WORK/serve_pi05.sbatch"
