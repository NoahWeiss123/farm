#!/bin/bash
# Base FFT-56k vs FFT-56k+LoRA on held-out episodes, per task. 1 GPU (inference).
# Heavy lifting in eval_fftlora_compare_inner.sh. JSON+NPZ → ~/farm-train/fftlora_analysis/.
#
#   sbatch eval_fftlora_compare.sbatch
#   SEED=0 NEPS=40 SPE=10 sbatch eval_fftlora_compare.sbatch
#   tail -f eval-fftlora-cmp-<jobid>.out

#SBATCH --job-name=eval-fftlora-cmp
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=01:15:00
#SBATCH --output=eval-fftlora-cmp-%j.out
# Short walltime + modest mem so it backfills into a GPU gap on the busy shared
# cluster (the eval is ~40 min; a 1h15 job fits far more backfill windows than 3h).
# Modest --mem (not the 960G default) so it backfills onto a spare GPU on the busy
# shared cluster instead of waiting for a half-node slot.

set -euo pipefail
WORK="$HOME/farm-train"
export SEED="${SEED:-0}" NEPS="${NEPS:-40}" SPE="${SPE:-10}"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_fftlora_compare_inner.sh"

echo "✓ eval-fftlora-compare job ${SLURM_JOB_ID} done — JSON+NPZ in $WORK/fftlora_analysis/"
