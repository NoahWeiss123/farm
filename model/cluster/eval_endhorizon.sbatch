#!/bin/bash
# Generic real-episode end-of-horizon eval — multiobject GSE model on an
# arbitrary dataset (set REPO_ID + TAG at submit time). 1 GPU, inference only.
#
#   sbatch --export=ALL,TAG=ood,REPO_ID=NoahWeiss/farm_uf850_bottle eval_endhorizon.sbatch
#   tail -f eval-eh-<jobid>.out      # JSON: ~/farm-train/eval-eh-<TAG>.json

#SBATCH --job-name=farm-eval-eh
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:35:00
#SBATCH --output=eval-eh-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/eval_endhorizon_inner.sh"

echo "✓ eval-eh job ${SLURM_JOB_ID} (TAG=${TAG:-ood}) done"
