#!/usr/bin/env bash
# One-time setup on the CS153 cluster login pod.
#
# Run this FROM the login pod (after `kubectl exec`-ing in), not your laptop.
# It clones openpi, registers the FARM configs by patching openpi's
# config.py, sets HF credentials, and leaves you ready to `sbatch`.
#
# Usage:
#   bash setup.sh <HF_TOKEN>
#
# Expects these files already staged alongside it under ~/farm-train
# (the README's "copy the launcher files" step does this):
#   patch_openpi_config.py        — adds LeRobotFarmDataConfig (+ pi0_fast LoRA)
#   patch_openpi_config_pi05.py   — adds pi05_farm_uf850 (full fine-tune)
#   train_pi05.sbatch             — the SLURM job
#   push_checkpoints.py           — background checkpoint → HF pusher
#
# After this, kick off training with:
#   sbatch ~/farm-train/train_pi05.sbatch
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash setup.sh <HF_TOKEN>" >&2
  exit 2
fi
HF_TOKEN_INPUT="$1"

# 1) Persistent workspace under $HOME (survives pod restarts).
WORK="$HOME/farm-train"
mkdir -p "$WORK"
cd "$WORK"

# 2) Clone openpi if not already there.
if [[ ! -d openpi ]]; then
  git clone --depth 1 https://github.com/Physical-Intelligence/openpi
fi

# 3) Register the FARM configs by patching openpi's config.py.
#    openpi reads its config registry from the _CONFIGS list in
#    src/openpi/training/config.py — there is no configs/ subpackage to
#    drop a file into — so we string-insert into that file. The patch
#    scripts are idempotent (sentinel-guarded) and order matters:
#    patch_openpi_config.py defines LeRobotFarmDataConfig, which the pi05
#    full-FT config reuses.
export OPENPI_CONFIG_PY="$WORK/openpi/src/openpi/training/config.py"
if [[ ! -f "$OPENPI_CONFIG_PY" ]]; then
  echo "ERROR: openpi config.py not found at $OPENPI_CONFIG_PY" >&2
  echo "       (did the clone in step 2 fail?)" >&2
  exit 1
fi

echo ">>> patching openpi config.py …"
python3 "$WORK/patch_openpi_config.py"
python3 "$WORK/patch_openpi_config_pi05.py"
# Register the LoRA variant too (opt-in; the sbatch still defaults to the
# full-FT pi05_farm_uf850). Harmless to register — it only adds a config
# name. Train it with CONFIG_NAME=pi05_farm_uf850_lora. The file is optional,
# so skip cleanly if it wasn't staged.
if [[ -f "$WORK/patch_openpi_config_pi05_lora.py" ]]; then
  python3 "$WORK/patch_openpi_config_pi05_lora.py"
fi
# Register the GSE variant (opt-in). Installs the GSE module + wiring
# (patch_openpi_gse.py needs openpi_gse.py staged beside it) then the config.
# Both idempotent + self-py_compile. Train it with train_pi05_gse.sbatch.
if [[ -f "$WORK/patch_openpi_gse.py" && -f "$WORK/openpi_gse.py" ]]; then
  python3 "$WORK/patch_openpi_gse.py" --openpi-root "$WORK/openpi"
  python3 "$WORK/patch_openpi_config_pi05_gse.py"
fi
# Mirror the per-step metrics line through logging so the loss reaches the
# SLURM log (openpi's tqdm redirect otherwise swallows it) and the /train
# dashboard can plot a loss curve. Benefits every config; idempotent.
if [[ -f "$WORK/patch_openpi_train_logging.py" ]]; then
  python3 "$WORK/patch_openpi_train_logging.py" --openpi-root "$WORK/openpi"
fi

# 4) Verify the patch is syntactically valid and the config name landed.
#    py_compile checks syntax WITHOUT importing openpi's heavy deps
#    (jax/torch aren't on the login pod) — so this is a cheap login-pod
#    sanity gate that catches a broken patch before we burn a GPU slot.
echo ">>> verifying patched config.py …"
python3 -m py_compile "$OPENPI_CONFIG_PY"
for name in '"pi05_farm_uf850"' 'class LeRobotFarmDataConfig'; do
  if ! grep -q "$name" "$OPENPI_CONFIG_PY"; then
    echo "ERROR: expected '$name' in config.py after patching — patch failed" >&2
    exit 1
  fi
done
echo "    ok: pi05_farm_uf850 + LeRobotFarmDataConfig registered, config.py compiles"
if grep -q '"pi05_farm_uf850_lora"' "$OPENPI_CONFIG_PY"; then
  echo "    ok: pi05_farm_uf850_lora (LoRA variant) also registered"
fi
if grep -q '"pi05_farm_uf850_gse"' "$OPENPI_CONFIG_PY"; then
  echo "    ok: pi05_farm_uf850_gse (GSE variant) also registered"
fi

# 5) Stash the HF token in a private env file the sbatch script sources.
#    File is chmod 600 so other students on shared storage can't read it.
ENVFILE="$WORK/.hf_env"
umask 077
cat > "$ENVFILE" <<EOF
export HF_TOKEN="$HF_TOKEN_INPUT"
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF
chmod 600 "$ENVFILE"
echo "wrote $ENVFILE (chmod 600)"

echo
echo "✓ setup complete"
echo "  workspace : $WORK"
echo "  openpi    : $WORK/openpi"
echo "  config    : pi05_farm_uf850 (π0.5 full fine-tune) registered"
echo "  hf env    : $ENVFILE (don't commit this)"
echo
echo "Next:"
echo "  cd $WORK && sbatch train_pi05.sbatch"
