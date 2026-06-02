#!/bin/bash
# Real-episode prediction eval of the BASE-INIT bottle LoRA (job 188), for the
# head-to-head vs the GSE-init LoRA: OOD (full) at 9999 + OOD checkpoint sweep
# + in-dist. 1 GPU. Runs in parallel with eval_lora_gse.sbatch.
#   sbatch eval_lora_base.sbatch ; tail -f eval-lora-base-<jobid>.out
#SBATCH --job-name=farm-eval-lora-base
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:10:00
#SBATCH --output=eval-lora-base-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_lora_runner.sh" base
echo "✓ eval-lora-base job ${SLURM_JOB_ID} done"
