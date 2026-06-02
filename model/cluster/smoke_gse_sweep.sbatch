#!/bin/bash
# GO/NO-GO smoke for the GSE expert sweep: the LARGEST config (79 specialized =
# 80 total experts, adapter rank 160) at batch 32 on ONE H100 — the binding
# memory constraint. If this fits, every smaller-expert variant fits.
#
# Verifies: (1) gemma builds with FARM_GSE_NUM_SPECIALIZED from env, (2)
# GSESVDWeightLoader SVD-inits 79 experts without a shape error, (3) NO OOM at
# batch 32 / 1 GPU, (4) loss is finite & healthy, (5) rough s/step for walltime
# planning. Runs ~120 steps (past JIT compile), pushes NOTHING, and deletes its
# throwaway checkpoint dir on exit.
#
# Submit:  sbatch smoke_gse_sweep.sbatch
# Look for: SMOKE_GSE_OK  (and SMOKE_SSTEP=…)

#SBATCH --job-name=smoke-gse-sweep
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=smoke-gse-sweep-%j.out

set -euo pipefail

WORK="$HOME/farm-train"
source "$WORK/.hf_env"
export CONFIG_NAME="pi05_farm_multiobject_gse_sweep"
export FARM_GSE_NUM_SPECIALIZED="${FARM_GSE_NUM_SPECIALIZED:-79}"
export EXP_NAME="smoke_gse_sweep_${SLURM_JOB_ID}"
export SMOKE_STEPS="${SMOKE_STEPS:-120}"

srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-workdir="$WORK/openpi" \
     --container-mounts="$HOME:$HOME" \
     bash -lc "
set -euo pipefail
export HOME='${HOME}'
export HF_TOKEN='${HF_TOKEN}'
export CONFIG_NAME='${CONFIG_NAME}'
export EXP_NAME='${EXP_NAME}'
export FARM_GSE_NUM_SPECIALIZED='${FARM_GSE_NUM_SPECIALIZED}'
export SMOKE_STEPS='${SMOKE_STEPS}'
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export FARM_AUG_LEVEL=heavy
export FARM_PROMPT_AUG=1

echo \">>> SMOKE: num_specialized=\$FARM_GSE_NUM_SPECIALIZED (total experts = 1 + that), batch 32, \$SMOKE_STEPS steps\"
apt-get -qq update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
pip install -q uv
cd \"\$HOME/farm-train/openpi\"
uv sync --frozen

uv run python -c \"import jax; print('jax devices:', jax.devices())\"

echo '>>> verifying FARM_GSE_NUM_SPECIALIZED propagates to the GSE config…'
uv run python -c \"
import os
from openpi.models import gemma
cfg = gemma.get_config('gemma_2b_gse')
ns = cfg.lora_configs['attn'].num_specialized
env = int(os.environ.get('FARM_GSE_NUM_SPECIALIZED','7'))
print(f'RESOLVED num_specialized={ns}  total_experts={ns+1}  adapter_rank={2+ns*2}  (env={env})')
assert ns == env, f'ENV VAR DID NOT PROPAGATE: got {ns}, want {env}'
print('ENV_WIRING_OK')
\"

NORM=\"\$HOME/farm-train/openpi/assets/\$CONFIG_NAME/NoahWeiss/farm_uf850_multiobject/norm_stats.json\"
if [[ -f \"\$NORM\" ]]; then
  echo \">>> norm stats present — skipping compute\"
else
  echo '>>> computing norm stats (one-time)…'
  uv run scripts/compute_norm_stats.py --config-name=\$CONFIG_NAME
fi

CKPT_DIR=\"\$HOME/farm-train/openpi/checkpoints/\$CONFIG_NAME/\$EXP_NAME\"
trap 'echo \">>> cleaning smoke checkpoints\"; rm -rf \"'\"\$CKPT_DIR\"'\" 2>/dev/null || true' EXIT

set +e
T0=\$(date +%s)
uv run python scripts/train.py \$CONFIG_NAME --exp-name=\"\$EXP_NAME\" --overwrite --no-wandb-enabled --num-train-steps=\$SMOKE_STEPS 2>&1 \
  | awk '{ print strftime(\"%s\"), \$0; fflush() }' \
  | tee /tmp/smoke_gse.log \
  | grep -E --line-buffered 'Step |RESOURCE_EXHAUSTED|OOM|Traceback|Error|error|assert|Killed'
RC=\${PIPESTATUS[0]}
T1=\$(date +%s)
set -e

echo \">>> train.py exit code: \$RC ; wall \$((T1-T0))s for \$SMOKE_STEPS steps (incl JIT compile)\"
# steady-state s/step from the two timestamped 'Step' log lines (e.g. 0 and 100)
python3 - <<'PY'
import re
ts=[]
for ln in open('/tmp/smoke_gse.log'):
    m=re.match(r'^(\d+)\s+Step (\d+):', ln)
    if m: ts.append((int(m.group(2)), int(m.group(1))))
ts=sorted(set(ts))
if len(ts)>=2:
    (s0,t0),(s1,t1)=ts[0],ts[-1]
    if s1>s0:
        print(f'SMOKE_SSTEP={(t1-t0)/(s1-s0):.3f}s  (from logged steps {s0}->{s1})')
print('logged steps:', [s for s,_ in ts])
PY
if [[ \$RC -eq 0 ]]; then echo 'SMOKE_GSE_OK'; else echo 'SMOKE_GSE_FAIL'; fi
exit \$RC
"

echo "✓ smoke job ${SLURM_JOB_ID} finished — GPU released"
