"""Add a π0.5 LoRA config (pi05_farm_bottle_lora) trained on the 100-episode
``NoahWeiss/farm_bottle_lora`` dataset.

Idempotent. Depends on ``LeRobotFarmDataConfig`` (inserted by
``patch_openpi_config.py`` — run that first; ``setup.sh`` does). Coexists with
the existing ``pi05_farm_uf850_lora`` (which trains on the 200-episode
``farm_uf850_bottle``) — this is the single-task, 100-episode sibling.

Differences from ``pi05_farm_uf850_lora``:
  * ``repo_id="NoahWeiss/farm_bottle_lora"`` (100 eps / 26,378 frames, the first
    100 of FARM dataset4, single task "Picking up the bottle and placing it on
    the box").
  * 10k steps (≈12 epochs at batch 32 over 26k frames) instead of 12k.
  * checkpoints every 2k (more selection granularity on the shorter run).
  * num_workers=32 — pair with --cpus-per-task=64 so the video-decoding data
    loader never starves a single H100.

Everything else mirrors the proven LoRA recipe: π0.5 backbone frozen, low-rank
adapters on the LLM + action expert, absolute actions, continuous state,
action_horizon=10 — so the same serve/eval path (model/eval_pi05.py) works
unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_BOTTLE_LORA_INSERTED"

# Shared model kwargs — must be identical in ``model=`` and in the
# ``freeze_filter`` instantiation so the freeze filter matches the
# actually-built parameter tree (same pattern as pi05_farm_uf850_lora).
_MODEL_KWARGS = (
    "pi05=True, action_horizon=10, discrete_state_input=False, "
    'paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"'
)

TRAIN_CONFIG_INSERT = f'''    #
    # PI05_FARM_BOTTLE_LORA_INSERTED — π0.5 LoRA on farm_bottle_lora (100 eps)
    #
    # Single-task, 100-episode sibling of pi05_farm_uf850_lora. Trains low-rank
    # adapters on the frozen π0.5 backbone + action expert. Fits on ONE H100
    # (no fsdp_devices) — submit with --gres=gpu:1 --cpus-per-task=64.
    TrainConfig(
        name="pi05_farm_bottle_lora",
        model=pi0_config.Pi0Config({_MODEL_KWARGS}),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_bottle_lora",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config({_MODEL_KWARGS}).get_freeze_filter(),
        # EMA off for LoRA (matches openpi's *_lora templates).
        ema_decay=None,
        # batch 32 fits a single 80GB H100. 26,378 frames / 32 ≈ 824 steps per
        # epoch → 10k steps ≈ 12 epochs. Checkpoints every 2k → step-2000…10000
        # for held-out selection (earlier often generalizes better — FINDINGS).
        batch_size=32,
        num_train_steps=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=10_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=2_000,
        keep_period=2_000,
        num_workers=32,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for pi05 bottle LoRA — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print(
            "ERROR: LeRobotFarmDataConfig not found in config.py.\n"
            "       Run patch_openpi_config.py first (setup.sh does this).",
            file=sys.stderr,
        )
        return 1
    anchor = '    TrainConfig(\n        name="pi05_libero"'
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find pi05_libero anchor", file=sys.stderr)
        return 1
    text = text[:idx] + TRAIN_CONFIG_INSERT + text[idx:]
    CFG.write_text(text)
    print(f"patched {CFG}")
    print("  + TrainConfig(name='pi05_farm_bottle_lora')  (π0.5 LoRA · farm_bottle_lora 100 eps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
