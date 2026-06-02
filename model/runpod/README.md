# model/runpod/ — RunPod serving path (isolated from the SLURM cluster code)

An **alternative to the CS153 SLURM cluster** for *serving* the π0.5 policy on a
rented GPU pod. The arm + RealSense cameras stay on your local machine running
`farm serve` + `model/eval_pi05.py`; only **policy inference** runs on the pod
(a Mac can't run π0.5 on GPU). This directory is kept separate so the RunPod
bootstrap doesn't get confused with the SLURM `sbatch` scripts in
`model/cluster/`.

> **Default is the cluster.** We currently serve on the CS153 cluster
> (`model/cluster/serve_pi05.sbatch`). RunPod is the fallback when the cluster
> queue is full. Validated working on an **RTX PRO 4500 Blackwell (sm_120)** —
> openpi's jaxlib can't emit sm_120 SASS, so XLA falls back to the CUDA-13
> driver's PTX JIT (one-time `ptxas does not support CC 12.0` warning, then fine).

## `pod_setup.sh`
Bootstraps a fresh pod: apt deps (ffmpeg), `uv`, clones openpi @ the cluster's
pinned commit (`c23745b5`), applies the FARM patches, `uv sync`, then **gates on
a real JAX GPU matmul**. It needs these patch files from `../cluster/` staged
beside it on the pod:
`patch_openpi_config.py`, `patch_openpi_gse.py`, `openpi_gse.py`,
`patch_openpi_config_pi05_gse_multiobject.py`.

## Using the RunPod path (laptop → pod)
```bash
POD=root@<ip>; PORT=<port>          # e.g. root@213.173.110.44  47609
# 1. stage + bootstrap
ssh $POD -p $PORT 'mkdir -p /root/farm-train'
scp -P $PORT ../cluster/patch_openpi_config.py ../cluster/patch_openpi_gse.py \
    ../cluster/openpi_gse.py ../cluster/patch_openpi_config_pi05_gse_multiobject.py \
    pod_setup.sh $POD:/root/farm-train/
ssh $POD -p $PORT 'cd /root/farm-train && bash pod_setup.sh'   # ~10 min; gates on JAX

# 2. checkpoint (public repo — note the step is 5999, not 6000)
ssh $POD -p $PORT 'cd /root/farm-train/openpi && uv run huggingface-cli download \
    NoahWeiss/farm_uf850_multiobject_gse_robust --include "step-5999/*" \
    --local-dir /root/farm-train/ckpt_mo_gse'

# 3. serve on the pod (:8000)
ssh $POD -p $PORT 'cd /root/farm-train/openpi && uv run python scripts/serve_policy.py \
    --port 8000 --default_prompt "Pick up the bottle and place it on the box" \
    policy:checkpoint --policy.config=pi05_farm_multiobject_gse \
    --policy.dir=/root/farm-train/ckpt_mo_gse/step-5999'

# 4. tunnel to the laptop, then drive the real arm locally
ssh -fNL 8000:localhost:8000 $POD -p $PORT
python model/eval_pi05.py --queue --policy-url ws://127.0.0.1:8000 --live   # CSV → test_logging/
```

Auth note: the pod uses key-based SSH; your key lives at `~/.ssh/id_ed25519`
(public key must be in the pod's `/root/.ssh/authorized_keys`).
