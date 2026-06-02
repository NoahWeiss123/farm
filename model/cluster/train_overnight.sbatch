#!/bin/bash
# Overnight 4-GPU run: three GSE fine-tunes that layer DOMAIN RANDOMIZATION and/or
# PROMPT-PARAPHRASE augmentation on the proven pi05_farm_uf850_gse recipe
# (batch 128, 3k steps ≈ 6.5 epochs, ~1.3h each on 4× H100, data-parallel/no-FSDP).
# All three run sequentially inside ONE job so the container builds + deps sync +
# norm-stats run ONCE (amortised). Together with the already-trained vanilla GSE
# (default aug, no prompt aug) they form a clean 2×2 ablation on
# (visual aug: default | heavy) × (prompt aug: off | on); the heavy+prompt cell
# ("robust") is the flagship and is trained FIRST so the best model lands earliest.
#
#   robust     : heavy aug + prompt aug → NoahWeiss/farm_uf850_pi05_gse_robust  (FLAGSHIP)
#   augonly    : heavy aug              → NoahWeiss/farm_uf850_pi05_gse_aug
#   promptonly : prompt aug             → NoahWeiss/farm_uf850_pi05_gse_prompt
#
# Why this is the right overnight bet: the dominant live-failure cause is a
# living-room→plywood-room DOMAIN SHIFT (see model/FINDINGS.md + env-mismatch
# notes). No architecture fixes a domain shift, but training-time domain
# randomization is its standard mitigation; GSE preserves π0.5's robust
# pretrained vision priors (full FT erased them). 4 GPUs only.
#
#   sbatch train_overnight.sbatch ; tail -f train-overnight-<jobid>.out

#SBATCH --job-name=farm-gse-robust
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=10:00:00
#SBATCH --output=train-overnight-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
source "$WORK/.hf_env"

# The heavy lifting is in a standalone inner script (no nested-quote escaping).
# It locates its own dir → derives HOME and sources .hf_env, so nothing needs
# exporting across the srun/container boundary here.
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash "$WORK/train_overnight_inner.sh"

echo "✓ overnight training job ${SLURM_JOB_ID} finished"
echo "  flagship : https://huggingface.co/NoahWeiss/farm_uf850_pi05_gse_robust"
echo "  ablations: ..._gse_aug  ..._gse_prompt"
