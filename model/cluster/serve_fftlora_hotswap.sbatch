#!/bin/bash
# Serve FFT-56k with LIVE per-object LoRA hot-swap (drives the /user page).
#
# One GPU holds the frozen FFT-56k base + all four per-object adapters
# (bottle/bear/duck/hat). The server (serve_policy_hotswap.py) routes the
# incoming task prompt to a skill and swaps only that skill's lora_a/lora_b
# leaves into the resident param tree — no base reload, no recompile. The
# /user orchestrator changes the prompt between steps, so the adapter follows
# the plan automatically.
#
# Submit:  sbatch serve_fftlora_hotswap.sbatch
#          ADAPTER_STEP=9000 sbatch serve_fftlora_hotswap.sbatch
#          SKILLS="bottle bear" sbatch serve_fftlora_hotswap.sbatch
# Logs:    tail -f /home/nhweiss/farm-train/serve-<jobid>.out
# Port:    8000 (relayed to the laptop by the daemon's two-hop tunnel)

#SBATCH --job-name=serve-fftlora-hotswap
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/home/nhweiss/farm-train/serve-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
CONFIG="${SERVE_CONFIG:-pi05_fftlora}"
ADAPTER_STEP="${ADAPTER_STEP:-9000}"    # step-9000 is the only checkpoint present in ALL four HF repos
SKILLS="${SKILLS:-bottle bear duck hat}"
HOTSWAP_PY="$WORK/serve_policy_hotswap.py"
# skill -> HF repo (the per-task LoRAs trained off frozen FFT-56k)
declare -A REPO=(
  [bottle]="NoahWeiss/farm_fftlora_bottle30"
  [bear]="NoahWeiss/farm_fftlora_bear30"
  [duck]="NoahWeiss/farm_fftlora_duck30"
  [hat]="NoahWeiss/farm_fftlora_hat30"
)

# HF token (only needed if a checkpoint must be downloaded).
[[ -f "$WORK/.hf_env" ]] && source "$WORK/.hf_env"

# ── Pre-flight (fail before burning container build / GPU minutes) ──────────
CFG_PY="$WORK/openpi/src/openpi/training/config.py"
if ! grep -q "$CONFIG" "$CFG_PY" 2>/dev/null; then
  echo "FATAL: config '$CONFIG' not registered in $CFG_PY" >&2
  echo "       apply: python model/cluster/patch_openpi_config_pi05_fftlora.py" >&2
  exit 1
fi
if [[ ! -f "$HOTSWAP_PY" ]]; then
  echo "FATAL: $HOTSWAP_PY not staged (copy model/cluster/serve_policy_hotswap.py to \$WORK)" >&2
  exit 1
fi

# Build the "skill=repo@step" handoff string for the container side.
SKILL_REPOS=""
for sk in $SKILLS; do
  [[ -n "${REPO[$sk]:-}" ]] || { echo "FATAL: no HF repo mapped for skill '$sk'" >&2; exit 1; }
  SKILL_REPOS="$SKILL_REPOS $sk=${REPO[$sk]}"
done

echo ">>> FFT-LoRA hot-swap serve · config=$CONFIG · step=$ADAPTER_STEP · skills:$SKILLS"

# ── Launch inside the NGC container (resolve ckpts, then serve) ─────────────
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd \"\$HOME/farm-train/openpi\"
pip install -q uv
uv sync --frozen

STEP='${ADAPTER_STEP}'
ADAPTERS=()
POLICY_DIR=''
for pair in ${SKILL_REPOS}; do
  sk=\"\${pair%%=*}\"; repo=\"\${pair#*=}\"
  base=\"\$HOME/farm-train/checkpoints/fftlora_\${sk}\"
  dir=\"\$base/step-\${STEP}\"
  if [[ ! -d \"\$dir/params\" ]]; then
    echo \">>> downloading \$repo step-\${STEP} → \$base\"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download \"\$repo\" --include \"step-\${STEP}/*\" --local-dir \"\$base\"
  fi
  if [[ ! -d \"\$dir/params\" ]]; then
    echo \"FATAL: no params for \$sk at \$dir after download\" >&2
    echo \"       check that \$repo has a step-\${STEP}/ dir (override with ADAPTER_STEP).\" >&2
    exit 1
  fi
  ADAPTERS+=( --adapter \"\${sk}=\${dir}\" )
  [[ -z \"\$POLICY_DIR\" ]] && POLICY_DIR=\"\$dir\"
done

echo '>>> serve_policy.py on :8000 (fftlora hot-swap)'   # cluster.py LAUNCHED marker
uv run python \"\$HOME/farm-train/serve_policy_hotswap.py\" \
    --config '${CONFIG}' \
    --policy-dir \"\$POLICY_DIR\" \
    \"\${ADAPTERS[@]}\" \
    --default-skill bottle \
    --port 8000
"
