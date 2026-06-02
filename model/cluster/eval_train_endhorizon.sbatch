#!/bin/bash
# End-of-horizon action-accuracy of the multiobject GSE policy on its TRAINING
# set: random middle frames, pred[H-1] vs state[t+H]. 1 GPU, inference only.
# Heavy lifting in eval_train_endhorizon_inner.sh (no quote escaping).
#
#   sbatch eval_train_endhorizon.sbatch
#   tail -f eval-train-endhorizon-<jobid>.out
#   # JSON: ~/farm-train/eval-train-endhorizon.json

#SBATCH --job-name=farm-eval-train-eh
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=eval-train-endhorizon-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_train_endhorizon_inner.sh"

echo "✓ eval-train-endhorizon job ${SLURM_JOB_ID} done — JSON in $WORK/eval-train-endhorizon.json"
