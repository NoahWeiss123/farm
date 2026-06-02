#!/bin/bash
# WITHIN-TRAINING per-episode eval: 15 episodes the bottle LoRAs trained on
# (farm_bottle_lora = multiobject[0:100]), 30 frames each, for GSE-init LoRA +
# base-init LoRA + GSE model on the SAME 15 episodes (seed-fixed). For the
# per-episode average + outlier analysis. 1 GPU.
#   sbatch eval_lora_fit15.sbatch ; tail -f eval-lora-fit15-<jobid>.out
#SBATCH --job-name=farm-eval-lora-fit15
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=eval-lora-fit15-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_lora_runner.sh" fit15
echo "✓ eval-lora-fit15 job ${SLURM_JOB_ID} done"
