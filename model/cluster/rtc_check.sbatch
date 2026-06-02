#!/bin/bash
# Validate Real-Time Chunking (PI-style seam-smoothing) on a chosen checkpoint —
# defaults to the overnight FLAGSHIP (GSE + heavy aug + prompt). Self-contained
# on the cluster (no websocket / port-forward): loads the policy and checks that
# RTC guidance pulls a new action chunk onto the previous one (smaller chunk-seam
# deviation = smoother motion), exercising the eager sampler, the jitted path,
# AND the served Policy.infer() path. ~7 min on 1 GPU.
#
#   sbatch rtc_check.sbatch                       # flagship
#   CHECK_CONFIG=pi05_farm_uf850 CHECK_CKPT='.../19999' sbatch rtc_check.sbatch
#
#SBATCH --job-name=rtc-check
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/home/nhweiss/farm-train/rtc-check-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
CONFIG="${CHECK_CONFIG:-pi05_farm_uf850_gse}"
CKPT="${CHECK_CKPT:-$WORK/openpi/checkpoints/pi05_farm_uf850_gse/farm_gse_robust_*/2999}"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-mounts="$HOME:$HOME" \
     --container-workdir="$WORK/openpi" \
     bash -lc "set -e; export HOME='$HOME'; export JAX_PLATFORMS=cuda; export TF_CPP_MIN_LOG_LEVEL=3;
       pip install -q uv 2>/dev/null; cd '$WORK/openpi';
       uv run python '$WORK/unit_test.py' --config '$CONFIG' --checkpoint-dir \"$CKPT\""
