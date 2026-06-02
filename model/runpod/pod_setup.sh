#!/bin/bash
# Bootstrap the openpi + FARM multiobject-GSE env on a fresh RunPod (Blackwell).
# Installs deps, clones openpi @ the cluster's pinned commit, applies the FARM
# patches, uv-syncs, then GATES on a real JAX GPU matmul (sm_120 / Blackwell).
# Does NOT download checkpoint/dataset yet — that waits until the JAX gate passes.
set -uo pipefail
WORK=/root/farm-train
OPENPI_COMMIT=c23745b5ad24e98f66967ea795a07b2588ed6c79
mkdir -p "$WORK"; cd "$WORK"
export DEBIAN_FRONTEND=noninteractive
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "[1/6] apt deps (ffmpeg, build-essential, git-lfs)"
apt-get -qq update >/dev/null 2>&1
apt-get install -y -qq ffmpeg build-essential git-lfs >/dev/null 2>&1 && log "  apt ok" || log "  apt FAILED"
ffmpeg -version 2>/dev/null | head -1

log "[2/6] uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version || { log "  uv install FAILED"; exit 1; }

log "[3/6] clone openpi @ ${OPENPI_COMMIT:0:10}"
if [ ! -d openpi/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi >/dev/null 2>&1 || { log "  clone FAILED"; exit 1; }
fi
( cd openpi && git checkout -q "$OPENPI_COMMIT" 2>/dev/null && log "  openpi at $(git rev-parse --short HEAD)" ) || log "  checkout note (continuing on HEAD)"

log "[4/6] apply FARM patches (config + GSE module + multiobject GSE config)"
export OPENPI_CONFIG_PY="$WORK/openpi/src/openpi/training/config.py"
python3 patch_openpi_config.py                         || { log "  patch_openpi_config FAILED"; exit 1; }
python3 patch_openpi_gse.py --openpi-root "$WORK/openpi" || { log "  patch_openpi_gse FAILED"; exit 1; }
python3 patch_openpi_config_pi05_gse_multiobject.py    || { log "  multiobject patch FAILED"; exit 1; }
python3 -m py_compile "$OPENPI_CONFIG_PY" && log "  config.py compiles"
grep -q pi05_farm_multiobject_gse "$OPENPI_CONFIG_PY" && log "  pi05_farm_multiobject_gse registered" || { log "  CONFIG MISSING"; exit 1; }

log "[5/6] uv sync (long: JAX + lerobot + deps) ..."
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen 2>&1 | tail -15
log "  uv sync returned ($?)"

log "[6/6] JAX GPU gate (Blackwell sm_120)"
uv run python - <<'PY' 2>&1 | tail -30
import jax, jax.numpy as jnp
print("jax", jax.__version__)
try:
    import jaxlib; print("jaxlib", jaxlib.__version__)
except Exception as e: print("jaxlib?", e)
devs = jax.devices(); print("devices:", devs)
gpu = any(d.platform == "gpu" for d in devs)
print("has_gpu:", gpu)
try:
    x = jnp.ones((4096, 4096)); y = float((x @ x).sum().block_until_ready())
    print("GPU_MATMUL_OK", y)
except Exception as e:
    print("GPU_MATMUL_FAILED", repr(e))
PY
log "SETUP DONE"
