#!/bin/bash
# FFT checkpoint generalization sweep: action accuracy of every FFT checkpoint on a
# fixed sample of multi-object (training-data) episodes. 1 GPU (inference). Heavy
# lifting in eval_fft_ckptgen_inner.sh. JSON+NPZ → ~/farm-train/fft_ckptgen/.
# Pair with `eval_fft_bench.sbatch fftonly` (held-out) + select_fft_base.py.
#
#   sbatch eval_fft_ckptgen.sbatch
#   SEED=0 NEPS=48 SPE=8 sbatch eval_fft_ckptgen.sbatch
#   tail -f eval-fft-ckptgen-<jobid>.out

#SBATCH --job-name=eval-fft-ckptgen
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=eval-fft-ckptgen-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
export SEED="${SEED:-0}" NEPS="${NEPS:-48}" SPE="${SPE:-8}"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_fft_ckptgen_inner.sh"

echo "✓ eval-fft-ckptgen job ${SLURM_JOB_ID} done — JSON+NPZ in $WORK/fft_ckptgen/"
