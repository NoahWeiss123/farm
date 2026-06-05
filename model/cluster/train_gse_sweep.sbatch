#!/bin/bash
# π0.5 GSE expert-count sweep — ONE variant per job, 1× H100.
#
# Recipe-matched to the FFT flagship (pi05_farm_multiobject_fft): batch 32,
# 56k steps, checkpoints 8k/16k/24k/32k/40k/48k/55999, cosine LR warmup 2k /
# peak 2.5e-5 / decay 56k -> 2.5e-6. GSE freezes the 2B backbone so it fits one
# GPU. The number of specialized experts is selected by the FARM_GSE_NUM_SPECIALIZED
# env var (read inside gemma.get_config); each job picks its own.
#
# Submit one per expert count, e.g.:
#   FARM_GSE_NUM_SPECIALIZED=79 EXP_TAG=e80 \
#     MODEL_REPO_ID=NoahWeiss/farm_uf850_multiobject_gse_e80 \
#     sbatch -J gse-e80 train_gse_sweep.sbatch
#   # default-expert LONG run (150k steps, decay still over 56k -> floor):
#   FARM_GSE_NUM_SPECIALIZED=7 EXP_TAG=e8 NUM_TRAIN_STEPS=150000 \
#     MODEL_REPO_ID=NoahWeiss/farm_uf850_multiobject_gse_e8 \
#     sbatch -J gse-e8 -t 5-00:00:00 train_gse_sweep.sbatch
#
# Partition note: batch 32 on ONE GPU is 32 samples/GPU — the same per-GPU
# footprint as the GSE flagship, which measured ~1.4-1.7 s/step. So a 56k run is
# ~22-27h (incl startup) and a plain run on 'small' (24h cap) would be SIGKILLed.
# Default is therefore 'medium' (5d cap, single-node OK). 56k runs use the 2-day
# header; the 150k run passes -t 5-00:00:00.
#
# Resume a walltime-capped run: resubmit with RESUME_FROM=<old EXP_NAME> and the
# SAME FARM_GSE_NUM_SPECIALIZED / MODEL_REPO_ID / EXP_TAG / NUM_TRAIN_STEPS (the
# sanity-map below re-checks the count; the schedule stays aligned since
# decay_steps is fixed at 56k regardless of --num-train-steps).
# Monitor: squeue -u $USER ; tail -f train-gse-sweep-<jobname>-<jobid>.out

#SBATCH --job-name=gse-sweep
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=train-gse-sweep-%x-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"

: "${FARM_GSE_NUM_SPECIALIZED:?set FARM_GSE_NUM_SPECIALIZED (e.g. 1,3,7,15,31,79)}"
: "${MODEL_REPO_ID:?set MODEL_REPO_ID (e.g. NoahWeiss/farm_uf850_multiobject_gse_e8)}"
: "${EXP_TAG:?set EXP_TAG (e.g. e8)}"

# Cross-check tag <-> expert count <-> repo so a transposed submit line can't
# silently train the wrong architecture into a mislabeled repo for 1-3 days.
case "$EXP_TAG" in
  e2)  _want_ns=1 ;;  e4)  _want_ns=3 ;;   e8)  _want_ns=7 ;;
  e16) _want_ns=15 ;; e32) _want_ns=31 ;;  e64) _want_ns=63 ;;  e80) _want_ns=79 ;;
  *)   _want_ns="" ;;
esac
if [[ -n "$_want_ns" && "$FARM_GSE_NUM_SPECIALIZED" != "$_want_ns" ]]; then
  echo "FATAL: EXP_TAG=$EXP_TAG expects FARM_GSE_NUM_SPECIALIZED=$_want_ns, got $FARM_GSE_NUM_SPECIALIZED" >&2
  exit 1
fi
case "$MODEL_REPO_ID" in
  *_"$EXP_TAG") : ;;
  *) echo "WARN: MODEL_REPO_ID ($MODEL_REPO_ID) does not end with _$EXP_TAG" >&2 ;;
esac
export FARM_GSE_NUM_SPECIALIZED MODEL_REPO_ID EXP_TAG
export CONFIG_NAME="pi05_farm_multiobject_gse_sweep"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
export RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" ]]; then
  export EXP_NAME="$RESUME_FROM"
else
  export EXP_NAME="farm_gse_sweep_${EXP_TAG}_${SLURM_JOB_ID}"
fi

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export MODEL_REPO_ID='${MODEL_REPO_ID}'
export EXP_NAME='${EXP_NAME}'
export EXP_TAG='${EXP_TAG}'
export CONFIG_NAME='${CONFIG_NAME}'
export NUM_TRAIN_STEPS='${NUM_TRAIN_STEPS}'
export RESUME_FROM='${RESUME_FROM}'
export FARM_GSE_NUM_SPECIALIZED='${FARM_GSE_NUM_SPECIALIZED}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Same robustness cell as the FFT/GSE flagships: heavy aug + prompt paraphrase.
export FARM_AUG_LEVEL=heavy
export FARM_PROMPT_AUG=1

echo \">>> GSE sweep: tag=\$EXP_TAG  num_specialized=\$FARM_GSE_NUM_SPECIALIZED  (total experts = 1 + that)\"
echo '>>> installing system ffmpeg…'
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null

pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

echo '>>> verifying jax sees 1 GPU…'
uv run python -c \"import jax; ds=jax.devices(); print('jax devices:', ds); assert len(ds) >= 1, f'want 1 GPU, got {len(ds)}'\"

NORM=\"\$HOME/farm-train/openpi/assets/\$CONFIG_NAME/NoahWeiss/farm_uf850_multiobject/norm_stats.json\"
if [[ -f \"\$NORM\" ]]; then
  echo \">>> norm stats present (\$NORM) — skipping the one-time 129k-frame compute\"
else
  echo '>>> computing norm stats (one-time, 129k frames)…'
  uv run scripts/compute_norm_stats.py --config-name=\$CONFIG_NAME
fi

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
mkdir -p \"\$CKPT_DIR\"
echo \">>> checkpoints → \$CKPT_DIR  (streamed to \$MODEL_REPO_ID)\"
uv run python \$HOME/farm-train/push_checkpoints.py \
    --checkpoint-dir \"\$CKPT_DIR\" \
    --repo-id \"\$MODEL_REPO_ID\" \
    --keep-period 8000 \
    --poll-interval 60 \
    > \$HOME/farm-train/push-gse-\${EXP_TAG}-\${SLURM_JOB_ID}.out 2>&1 &
PUSHER_PID=\$!
trap 'kill -TERM \$PUSHER_PID 2>/dev/null || true; wait \$PUSHER_PID 2>/dev/null || true' EXIT

echo \">>> starting GSE training (1 GPU, batch 32, experts=\$FARM_GSE_NUM_SPECIALIZED, steps=\${NUM_TRAIN_STEPS:-56000})…\"
if [[ -n \"\$RESUME_FROM\" ]]; then
  echo \"resuming from \$RESUME_FROM\"
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --resume --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}
else
  uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled \${NUM_TRAIN_STEPS:+--num-train-steps=\$NUM_TRAIN_STEPS}
fi

echo '>>> draining checkpoint pusher…'
kill -TERM \$PUSHER_PID 2>/dev/null || true
wait \$PUSHER_PID 2>/dev/null || true
echo '>>> final checkpoints:'
ls -1 \"\$CKPT_DIR\" 2>/dev/null || true
"

echo "✓ GSE-sweep job ${SLURM_JOB_ID} (${EXP_TAG}, experts=${FARM_GSE_NUM_SPECIALIZED}) finished — GPU released"
echo "model: https://huggingface.co/${MODEL_REPO_ID}"
