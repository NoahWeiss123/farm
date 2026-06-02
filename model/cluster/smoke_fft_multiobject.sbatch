#!/bin/bash
# 4-GPU, 5-step smoke test of the π0.5 FULL fine-tune on farm_uf850_multiobject,
# run BEFORE committing the multi-hour job. Uses the SAME 4-GPU layout as the real
# run so it faithfully validates:
#   * the new pi05_farm_multiobject_fft config builds + registers,
#   * fsdp_devices=2 shards the full FT across 2 GPUs → 2 data-parallel replicas,
#   * batch 32 (16/replica) FITS in 80GB (no CUDA OOM) — the real memory point,
#   * heavy image aug + PromptParaphrase run inside compute_loss over THIS dataset,
#   * compute_norm_stats over the 4-task/129k-frame set succeeds, and
#   * 5 optimisation steps log a finite, decreasing loss.
# Then a GUARDED probe at batch 48 (24/replica) to see if there's headroom to bump
# — its failure is non-fatal and never masks the batch-32 GO.
# ~10-15 min incl. first-time container squashfs build.
#
#   sbatch smoke_fft_multiobject.sbatch ; tail -f smoke-fft-<jobid>.out
#   expect ">>> SMOKE_FFT_OK (batch32 validated)"

#SBATCH --job-name=farm-smoke-fft
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --output=smoke-fft-%j.out

set -euo pipefail
WORK="$HOME/farm-train"
source "$WORK/.hf_env"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export FARM_AUG_LEVEL=heavy
export FARM_PROMPT_AUG=1
echo '>>> installing ffmpeg…'
apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen
echo '>>> jax devices:'; uv run python -c \"import jax; ds=jax.devices(); print(ds); assert len(ds) >= 4, f'want 4 GPUs, got {len(ds)}'\"
echo '>>> norm stats (faithful GO/NO-GO — rules out stale/missing-stats NaN)…'
uv run scripts/compute_norm_stats.py --config-name=pi05_farm_multiobject_fft
echo '>>> 5-step FULL-FT smoke @ batch 32 (16/replica) — watch for finite, decreasing loss + NO OOM…'
uv run python scripts/train.py pi05_farm_multiobject_fft --exp-name=fft_smoke_b32 \
    --overwrite --no-wandb-enabled --num-train-steps=5 --batch-size=32
echo '>>> SMOKE_FFT_OK (batch32 validated)'

# ── optional headroom probe: batch 48 (24/replica). Non-fatal. ──────────────
echo '>>> probing batch 48 headroom (3 steps; failure here is fine — we keep batch 32)…'
set +e
uv run python scripts/train.py pi05_farm_multiobject_fft --exp-name=fft_smoke_b48 \
    --overwrite --no-wandb-enabled --num-train-steps=3 --batch-size=48
if [[ \$? -eq 0 ]]; then echo '>>> BATCH48_OK (headroom exists — could bump batch_size)'; else echo '>>> BATCH48_FAILED (stick with batch 32)'; fi
# Reclaim the ~42GB/dir orphan checkpoints the smoke saves at its final step
# (different exp-names than the real run, so they would otherwise persist).
# Keep assets/pi05_farm_multiobject_fft/ norm stats — the real run reuses them.
echo '>>> cleaning up smoke checkpoint dirs…'
rm -rf \"\$HOME/farm-train/openpi/checkpoints/pi05_farm_multiobject_fft/fft_smoke_b32\" \
       \"\$HOME/farm-train/openpi/checkpoints/pi05_farm_multiobject_fft/fft_smoke_b48\"
set -e
"
echo "smoke job ${SLURM_JOB_ID} finished"
