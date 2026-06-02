#!/bin/bash
# Extract LoRA adapter tensors from every trained task-LoRA into compact .npz files
# (~tens of MB each) for the local vector analysis. 1 GPU (only for the openpi
# import; restore_params + numpy is CPU). Output → ~/farm-train/lora_vectors/.
#
#   sbatch extract_lora_vectors.sbatch ; then kubectl cp the .npz dir to the laptop
#   and run model/analyze_lora_vectors.py on it.

#SBATCH --job-name=extract-lora
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=extract-lora-%j.out
# CPU-ONLY (no --gres): adapter extraction is restore_params + numpy, no GPU. CPU
# cores aren't GPU-contended, so this schedules immediately even when GPU jobs are
# backlogged for days. JAX_PLATFORMS=cpu is set inside.

set -euo pipefail
WORK="$HOME/farm-train"
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -uo pipefail
export HOME='${HOME}'
export JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3
pip install -q uv; cd \"\$HOME/farm-train/openpi\"; uv sync --frozen
OUT=\"\$HOME/farm-train/lora_vectors\"; mkdir -p \"\$OUT\"
CKBASE=\"\$HOME/farm-train/openpi/checkpoints/pi05_fftlora\"
# MATCHED step across all 6 LoRAs (9000 = max common step; bottle30/bear30/hat30 hit
# the 4h walltime before 11999, all fully converged by 9000, loss ~3e-4) → the vector
# comparison is at an identical step for every task. Override with STEP=.
STEP=\"\${STEP:-9000}\"
for task in bottle30 bottle100 bear30 duck30 hat30 bottle30s1; do
  ck=\"\$CKBASE/fftlora_\$task/\$STEP\"
  if [ ! -d \"\$ck/params\" ]; then echo \"!! no step-\$STEP for \$task — skipping\"; continue; fi
  echo \"===== extracting \$task from \$ck =====\"
  uv run python \"\$HOME/farm-train/extract_lora_adapters.py\" --checkpoint-dir \"\$ck\" --label \"\$task\" --out \"\$OUT/\$task.npz\" || echo \"!! extract failed for \$task\"
done
echo '>>> EXTRACT DONE'; ls -la \"\$OUT\"
"
echo "extract-lora job ${SLURM_JOB_ID} done — vectors in $WORK/lora_vectors/"
