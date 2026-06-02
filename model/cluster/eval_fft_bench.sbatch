#!/bin/bash
# FFT-multiobject analysis eval: sweep every FFT checkpoint (clean + domain-shift
# robust) on eval_bench, then eval the 3 comparison techniques on the same bench
# for a self-consistent 4-way head-to-head. 1 GPU (inference only). Heavy lifting
# in eval_fft_bench_inner.sh (no quote escaping). JSONs → ~/farm-train/fft_analysis/.
#
#   sbatch eval_fft_bench.sbatch            # FFT sweep + baselines (full run)
#   sbatch eval_fft_bench.sbatch fftonly    # FFT sweep only (quick interim check)
#   FFT_EXP=farm_fft_multiobject_robust_<jobid> sbatch eval_fft_bench.sbatch
#   tail -f eval-fft-bench-<jobid>.out

#SBATCH --job-name=farm-eval-fft
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=eval-fft-bench-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
# FFT_EXP is consumed by the inner script; export so it crosses the srun boundary.
export FFT_EXP="${FFT_EXP:-farm_fft_multiobject_robust_406}"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_fft_bench_inner.sh" "$@"

echo "✓ eval-fft-bench job ${SLURM_JOB_ID} done — JSONs in $WORK/fft_analysis/"
