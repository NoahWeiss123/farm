"""Add the π0.5 FULL fine-tune config (pi05_farm_uf850) to openpi's config.py.

Idempotent. Depends on ``LeRobotFarmDataConfig`` already being present
(inserted by ``patch_openpi_config.py`` — run that one first; ``setup.sh``
does this for you). Coexists with the pi0_fast config; both stay registered.

This registers a **full fine-tune**, not LoRA. ``Pi0Config(pi05=True)`` with
its default (non-``_lora``) variants makes the entire π0.5 model trainable —
PaliGemma 2B backbone + Gemma 300M action expert — and there is no
``freeze_filter``, so every parameter updates.

Why a full fine-tune here:
  * 200 episodes / 59,183 frames / 2 tasks (~33 min @ 30 fps).
  * Full FT has more capacity than LoRA to move the policy onto a new
    embodiment (UF850, 6-joint + gripper) and a new visual scene. With
    only 2 tasks, overfitting is held in check by the cosine LR decay and
    a 20k-step budget (~11 epochs) plus EMA — ample for a 2-task dataset.

Memory: a full fine-tune of π0.5 (~3.3 B params) does not fit on a single
80 GB H100 at a useful batch size — Adam state alone is ~40 GB. So
``fsdp_devices=2`` shards params + grads + optimizer state across 2 GPUs.
The sbatch requests 8 H100s (the whole node), so there are 4 such FSDP
groups → 4 data-parallel replicas (same global batch, ~4× single-replica
throughput). If you still hit CUDA OOM, drop ``batch_size`` 32 → 16 (and,
if needed, lower ``peak_lr``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Path to openpi's training config. setup.sh exports OPENPI_CONFIG_PY; the
# literal default is the canonical cluster location for user `nhweiss`.
CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_UF850_INSERTED"

TRAIN_CONFIG_INSERT = '''    #
    # PI05_FARM_UF850_INSERTED — π0.5 FULL fine-tune for FARM UF850
    #
    # 200 episodes / 59,183 frames / 2 tasks (~33 min @ 30 fps).
    # FULL fine-tune: default (non-LoRA) variants + no freeze_filter, so
    # the whole π0.5 model trains. fsdp_devices=2 shards the ~3.3B-param
    # model across 2 GPUs (full FT won't fit on one); the sbatch requests
    # 8 H100s (whole node), so there are 4 such FSDP groups → 4 data-parallel
    # replicas (~4× single-replica throughput, same global batch). batch_size
    # =32; drop to 16 if OOM.
    #
    # 59,183 frames / batch 32 ≈ 1,850 steps/epoch → 20k steps ≈ 11
    # epochs — ample for a 2-task dataset. At full-FT throughput on 8× H100
    # (4 data-parallel × 2-way FSDP) that's ~1.5–2h wall time IF the 64-worker
    # data-loader keeps up; intermediate checkpoints (5k/10k/15k) let you
    # stop earlier.
    TrainConfig(
        name="pi05_farm_uf850",
        model=pi0_config.Pi0Config(
            pi05=True,
            # Do NOT override action_dim. π0.5 works in its 32-dim universal
            # action space; LiberoInputs pads our 7-DoF state/action up to it
            # and LiberoOutputs slices back to [:, :7]. Leaving action_dim at
            # its default (matching openpi's own pi05_libero) keeps the
            # pretrained action head from pi05_base loadable — hard-coding 7
            # would shape-mismatch those projections.
            action_horizon=10,
            # Continuous proprioceptive state (not tokenized) — matches
            # pi05_libero; correct for a continuous-joint arm like the UF850.
            discrete_state_input=False,
            # No *_lora variants → default gemma_2b + gemma_300m ⇒ full FT.
        ),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_bottle",
            base_config=DataConfig(prompt_from_task=True),
            # π0.5 trains on ABSOLUTE actions: openpi's pi05_libero sets
            # extra_delta_transform=False (π0 used delta). Our recorded
            # actions are already absolute next-state joint targets, so we
            # disable the delta wrap to match π0.5's convention exactly.
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # No freeze_filter ⇒ every parameter is trainable (full fine-tune).
        # EMA on — the full-FT convention (LoRA configs set ema_decay=None).
        ema_decay=0.999,
        # Shard the model across 2 GPUs (full FT won't fit on one). With 8
        # GPUs total (the whole node) that's 4 FSDP groups → 4 data-parallel
        # replicas. The global batch (64) is split across the 4 replicas
        # (16 each); GPU scaling itself is LR-neutral — the LR bump below is
        # only because the global batch grew 32→64.
        fsdp_devices=2,
        # Global batch 64 across 8 GPUs (4 DP replicas) = 16 samples/replica
        # — the exact per-GPU footprint the 4-GPU/batch-32 run already proved
        # fits. Bigger batch = less-noisy gradients for π0.5's flow-matching
        # objective (π0.5 itself trains at batch 256) and far better MFU than
        # batch 32 (which would give a tiny 8 samples/replica on 8 GPUs).
        # 20k steps × 64 / 59,183 ≈ 21.6 epochs; checkpoint selection
        # (5k/10k/15k/20k) picks the best and guards against overfit.
        batch_size=64,
        # 64 dataloader workers decode the base+wrist h264 frames in parallel
        # — sized to feed 8 GPUs (4 DP replicas) without starving them. The
        # node has 224 CPUs and the sbatch grabs 128, so ~2 CPUs/worker. An
        # earlier 4-GPU/12-worker run was data-bound (~1 it/s, jittery); this
        # is the fix for the 8-GPU scale-up.
        num_workers=64,
        num_train_steps=20_000,
        # LR scaled from openpi's pi05_libero recipe (peak 5e-5 @ batch 256)
        # to our batch 64 → 3.5e-5 peak (√2 above the batch-32 value of
        # 2.5e-5, the standard sqrt batch-scaling rule). π0.5 favors a long
        # warmup, so 2k steps (10%) before peak — protects the pretrained
        # features during early full-FT updates. Cosine decay to 3.5e-6 for
        # stable convergence on a small 2-task set (pi05_libero holds LR flat,
        # but it trains on far more data; decay is safer at our scale).
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3.5e-5,
            decay_steps=20_000,
            decay_lr=3.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Save every 5k, keep them all ⇒ step 5k,10k,15k,20k on disk
        # (4 ckpts). Keep push_checkpoints.py --keep-period in sync.
        save_interval=5_000,
        keep_period=5_000,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for pi05 — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print(
            "ERROR: LeRobotFarmDataConfig not found in config.py.\n"
            "       Run patch_openpi_config.py first (setup.sh does this).",
            file=sys.stderr,
        )
        return 1
    # Insert right before pi05_libero (an existing pi05 anchor in openpi).
    anchor = '    TrainConfig(\n        name="pi05_libero"'
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find pi05_libero anchor", file=sys.stderr)
        return 1
    text = text[:idx] + TRAIN_CONFIG_INSERT + text[idx:]
    CFG.write_text(text)
    print(f"patched {CFG}")
    print("  + TrainConfig(name='pi05_farm_uf850')  (π0.5 FULL fine-tune)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
