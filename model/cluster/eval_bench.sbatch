#!/bin/bash
# Benchmark eval: FFT vs LoRA vs multiobject-GSE on eval_bench (15 eps, 5 tasks),
# clean + domain-shift robustness → eval-{clean,robust}-{full,lora,multiobject}.json.
# 1 GPU (inference only). Heavy lifting in eval_bench_inner.sh (no quote escaping).
#
#   sbatch eval_bench.sbatch              # all three
#   sbatch eval_bench.sbatch lora multiobject
#   tail -f eval-bench-<jobid>.out

#SBATCH --job-name=farm-eval-bench
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=eval-bench-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_bench_inner.sh" "$@"

echo "✓ eval-bench job ${SLURM_JOB_ID} done — JSONs in $WORK/eval-{clean,robust}-*.json"
