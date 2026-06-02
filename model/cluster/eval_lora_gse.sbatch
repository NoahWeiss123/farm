#!/bin/bash
# Real-episode prediction eval of the GSE-INIT bottle LoRA (job 383):
# OOD + in-dist (full, +figures) at step-9999 + OOD checkpoint sweep. 1 GPU.
#   sbatch eval_lora_gse.sbatch ; tail -f eval-lora-gse-<jobid>.out
#SBATCH --job-name=farm-eval-lora-gse
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:10:00
#SBATCH --output=eval-lora-gse-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_lora_runner.sh" gse
echo "✓ eval-lora-gse job ${SLURM_JOB_ID} done"
