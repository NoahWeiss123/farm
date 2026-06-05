#!/bin/bash
# train_gse_off_fft.sbatch — π0.5 GSE e8 fine-tune INITIALIZED OFF the finished
# 56k FULL fine-tune (…_fft_robust step-55999), 8k steps, 1× H100.
#
# This is the GSE-off-base sweep's sibling, but the GSESVDWeightLoader source is
# the local 56k FFT checkpoint instead of gs://…/pi05_base — so the frozen 2B
# backbone is the FFT's tuned backbone, the GSE attention experts SVD-init off
# THAT spectrum, and the 300M action expert resumes from the FFT's fit weights.
#
# GENUINE HELD-OUT DESIGN. The FFT (the init) trained on ALL 424 episodes, so to
# get a TRUE held-out signal we train THIS GSE fine-tune on the per-task HEAD
# episodes only (FARM_EP_RANGE below = the eval's TRAIN_RANGES), leaving the
# per-task TAILS (HELDOUT_RANGES = 239:299,339:349,374:384,414:424) genuinely
# unseen by the GSE step. The frozen FFT backbone is identical for both splits,
# so comparing GSE-off-FFT@step-k to its init FFT@55999 isolates the effect of
# the 8k GSE continuation: on heads = continued fit, on tails = generalization of
# a continuation that never trained on them. Full-set norm stats are kept
# (shared normalization, matching the FFT init — see patch_openpi_epfilter.py).
#
# PROVENANCE. step-55999/params is the FFT's EMA-served tree (FFT config sets
# ema_decay=0.999; openpi writes EMA into params/). The separately-running FFT
# baseline eval restores the SAME step-55999/params, so init source and baseline
# are byte-identical. GSE serves raw params (ema_decay=None) — note this EMA→raw
# difference when comparing (it is the GSE convention, not a recipe slip).
#
# Question: does stacking GSE capacity on a converged full FT lower joint error
# on the heads it keeps training, and does it help or hurt the held-out tails it
# never sees? Config: pi05_farm_multiobject_gse_off_fft (8k cosine, dense 1k
# checkpoints). See patch_openpi_config_pi05_gse_off_fft.py.
#
# PREREQUISITES (do these on the login pod BEFORE sbatch — the sbatch verifies):
#   1. Stage FFT init params:  $HOME/farm-train/fft_init/step-55999/params
#        (snapshot_download step-55999/* from NoahWeiss/farm_uf850_multiobject_fft_robust)
#   2. Register the config:    python $HOME/farm-train/patch_openpi_config_pi05_gse_off_fft.py
#   (Norm stats are auto-staged below from the gse_sweep config's assets.)
#
# EVAL (after training): override eval_curve STEPS to this run's dense ladder —
#   STEPS="1000 2000 3000 4000 5000 6000 7000 7999"  (the default 8k..55999 won't match).
#
# 8k GSE steps at ~1.4-1.7 s/step ≈ 3.5-4h + startup/JIT → fits 'small' (24h).
# Submit:
#   sbatch -J gse-off-fft train_gse_off_fft.sbatch
# Resume a capped/cancelled run (same EXP_NAME):
#   RESUME_FROM=farm_gse_off_fft_e8_<oldjobid> sbatch -J gse-off-fft train_gse_off_fft.sbatch
# Monitor: squeue -u $USER ; tail -f train-gse-off-fft-<jobname>-<jobid>.out

#SBATCH --job-name=gse-off-fft
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=train-gse-off-fft-%x-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"

export CONFIG_NAME="pi05_farm_multiobject_gse_off_fft"
export FARM_GSE_NUM_SPECIALIZED="${FARM_GSE_NUM_SPECIALIZED:-7}"   # e8 = 8 total experts
export EXP_TAG="e8"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-8000}"
export FFT_INIT="$WORK/fft_init/step-55999/params"
export NORM_SRC="$WORK/openpi/assets/pi05_farm_multiobject_gse_sweep/NoahWeiss/farm_uf850_multiobject/norm_stats.json"
# Per-task HEAD episodes only (= eval TRAIN_RANGES) → the TAILS stay held out
# from this GSE fine-tune. Explicit episode list (data_loader's FARM_EP_RANGE
# else-branch parses a comma list of ints — no patch needed). 334 of 424 eps.
export FARM_EP_RANGE="$(python3 -c "print(','.join(str(e) for r in [(0,239),(299,339),(349,374),(384,414)] for e in range(*r)))")"
export RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" ]]; then
  export EXP_NAME="$RESUME_FROM"
else
  export EXP_NAME="farm_gse_off_fft_e8_${SLURM_JOB_ID}"
fi

# Guard the e8 expert count (this config is e8-only).
if [[ "$FARM_GSE_NUM_SPECIALIZED" != "7" ]]; then
  echo "FATAL: this config is e8 (FARM_GSE_NUM_SPECIALIZED=7); got $FARM_GSE_NUM_SPECIALIZED" >&2
  exit 1
fi

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export CONFIG_NAME='${CONFIG_NAME}'
export EXP_NAME='${EXP_NAME}'
export EXP_TAG='${EXP_TAG}'
export NUM_TRAIN_STEPS='${NUM_TRAIN_STEPS}'
export RESUME_FROM='${RESUME_FROM}'
export FARM_GSE_NUM_SPECIALIZED='${FARM_GSE_NUM_SPECIALIZED}'
export FFT_INIT='${FFT_INIT}'
export NORM_SRC='${NORM_SRC}'
export FARM_EP_RANGE='${FARM_EP_RANGE}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Same robustness cell as the FFT/GSE flagships the init came from.
export FARM_AUG_LEVEL=heavy
export FARM_PROMPT_AUG=1

_NEP=\$(echo \"\$FARM_EP_RANGE\" | tr ',' '\\n' | grep -c .)
echo \">>> GSE-off-FFT: config=\$CONFIG_NAME  experts=8 (num_specialized=\$FARM_GSE_NUM_SPECIALIZED)  steps=\$NUM_TRAIN_STEPS\"
echo \">>> heads-only training: FARM_EP_RANGE = \$_NEP episodes (tails 239:299,339:349,374:384,414:424 held out)\"
if [[ \"\$_NEP\" != \"334\" ]]; then echo \"FATAL: expected 334 head episodes, got \$_NEP\" >&2; exit 1; fi

echo '>>> verifying config is registered…'
if ! grep -q 'pi05_farm_multiobject_gse_off_fft' \"\$HOME/farm-train/openpi/src/openpi/training/config.py\"; then
  echo 'FATAL: config not registered — run patch_openpi_config_pi05_gse_off_fft.py on the login pod first.' >&2
  exit 1
fi

echo '>>> verifying FFT init params are staged (non-empty + orbax manifest + size floor)…'
if [[ ! -d \"\$FFT_INIT\" ]] || [[ -z \"\$(ls -A \"\$FFT_INIT\" 2>/dev/null)\" ]]; then
  echo \"FATAL: FFT init params missing/empty at \$FFT_INIT — stage step-55999 first (see header).\" >&2
  exit 1
fi
if [[ ! -f \"\$FFT_INIT/manifest.ocdbt\" ]]; then
  echo \"FATAL: \$FFT_INIT has no manifest.ocdbt — partial/corrupt orbax checkpoint; re-stage.\" >&2
  exit 1
fi
_FFT_KB=\$(du -sk \"\$FFT_INIT\" 2>/dev/null | cut -f1)
if [[ -z \"\$_FFT_KB\" || \"\$_FFT_KB\" -lt 4000000 ]]; then
  echo \"FATAL: FFT init only \${_FFT_KB:-0} KB (< 4 GB floor) — partial download; re-stage.\" >&2
  exit 1
fi
echo \"    FFT init OK: \$(du -sh \$FFT_INIT 2>/dev/null | cut -f1) at \$FFT_INIT\"

echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

echo '>>> verifying jax sees 1 GPU…'
uv run python -c \"import jax; ds=jax.devices(); print('jax devices:', ds); assert len(ds) >= 1, f'want 1 GPU, got {len(ds)}'\"

# Stage norm stats into the new config's assets (skip the one-time 129k-frame
# compute — same dataset+normalization as the gse_sweep config).
NORM_DST=\"\$HOME/farm-train/openpi/assets/\$CONFIG_NAME/NoahWeiss/farm_uf850_multiobject/norm_stats.json\"
if [[ -f \"\$NORM_DST\" ]]; then
  echo \">>> norm stats present (\$NORM_DST)\"
elif [[ -f \"\$NORM_SRC\" ]]; then
  mkdir -p \"\$(dirname \"\$NORM_DST\")\"
  cp \"\$NORM_SRC\" \"\$NORM_DST\"
  echo \">>> staged norm stats from gse_sweep → \$NORM_DST\"
else
  echo '>>> norm stats not found to copy — computing (one-time, 129k frames)…'
  uv run scripts/compute_norm_stats.py --config-name=\$CONFIG_NAME
fi

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo \">>> checkpoints → \$CKPT_DIR (LOCAL only; eval reads them locally)\"

echo \">>> starting GSE-off-FFT training (1 GPU, batch 32, e8, steps=\$NUM_TRAIN_STEPS)…\"
if [[ -n \"\$RESUME_FROM\" ]]; then
  echo \"resuming from \$RESUME_FROM\"
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --resume --no-wandb-enabled --num-train-steps=\$NUM_TRAIN_STEPS
else
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled --num-train-steps=\$NUM_TRAIN_STEPS
fi

echo '>>> final checkpoints:'
ls -1 \"\$CKPT_DIR\" 2>/dev/null || true
"

echo "✓ GSE-off-FFT job ${SLURM_JOB_ID} (e8, init off 56k FFT, heads-only) finished — GPU released"
echo "checkpoints: $HOME/farm-train/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
