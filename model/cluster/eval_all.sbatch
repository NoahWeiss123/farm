#!/bin/bash
# Run clean-fit (eval_offline.py) + domain-shift robustness (eval_robust.py)
# evals for a set of models on ONE GPU, writing one JSON pair per model under
# ~/farm-train. Re-runnable as checkpoints land:
#
#   MODELS="gse_robust" sbatch eval_all.sbatch          # just the flagship
#   sbatch eval_all.sbatch                              # all five (default)
#
# Default set: full (deployed full-FT), gse (vanilla GSE), and the three new
# augmentation variants. Checkpoints resolve local-first (shared Weka home),
# else download step-N from the HF repo.

#SBATCH --job-name=farm-eval-all
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=eval-all-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
source "$WORK/.hf_env"
MODELS="${MODELS:-full gse gse_robust gse_aug gse_prompt}"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_all_inner.sh" $MODELS

echo "✓ eval job ${SLURM_JOB_ID} finished — eval-clean-*.json / eval-robust-*.json in $WORK"
