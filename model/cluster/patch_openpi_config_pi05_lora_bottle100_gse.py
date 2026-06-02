"""Add ``pi05_farm_bottle_lora_gse`` — a π0.5 LoRA fine-tune that initializes off
the **fine-tuned GSE multiobject checkpoint** (step-5999) instead of base π0.5.

Identical recipe to ``pi05_farm_bottle_lora`` (same 100-episode bottle dataset =
``NoahWeiss/farm_bottle_lora`` ≡ ``farm_uf850_multiobject[:100]``, same
gemma_2b_lora / gemma_300m_lora adapters, batch 32, 10k steps, save every 2k) —
the ONLY change is the weight loader:

    weight_loaders.CheckpointWeightLoader("…/pi05_base/params")   # the old one
    weight_loaders.GSEMergeWeightLoader("…/farm_gse_multiobject_robust_190/5999/params")  # this

``GSEMergeWeightLoader`` (patch_openpi_gse_merge.py) merges the GSE checkpoint's
generalized/specialized attention adapters + FFN LoRA into the dense base, so the
frozen backbone the LoRA trains on top of IS the multiobject-specialized policy,
not generic π0.5. The point: the first 100 episodes are near-identical bottle
motions, so starting from a checkpoint that already mastered the multiobject
(incl. bottle) task should let the LoRA specialize faster / cleaner than starting
from base — the head-to-head against pi05_farm_bottle_lora is the experiment.

Idempotent. Depends on LeRobotFarmDataConfig (patch_openpi_config.py) and
GSEMergeWeightLoader (patch_openpi_gse_merge.py) — apply both first.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_BOTTLE_LORA_GSE_INSERTED"

# GSE multiobject checkpoint to LoRA off of (step-5999, the flagship robust run).
GSE_PARAMS = os.environ.get(
    "GSE_INIT_PARAMS",
    "/home/nhweiss/farm-train/openpi/checkpoints/pi05_farm_multiobject_gse/farm_gse_multiobject_robust_190/5999/params",
)

# LoRA adapters on the (now GSE-merged) backbone + action expert — same as the
# base-init sibling so the only variable is the initialization.
_MODEL_KWARGS = (
    "pi05=True, action_horizon=10, discrete_state_input=False, "
    'paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"'
)

TRAIN_CONFIG_INSERT = f'''    #
    # PI05_FARM_BOTTLE_LORA_GSE_INSERTED — π0.5 LoRA initialized off the GSE
    # multiobject checkpoint (not base π0.5). Same 100-ep bottle recipe as
    # pi05_farm_bottle_lora; only the weight_loader differs. 1× H100, --gres=gpu:1
    # --cpus-per-task=64.
    TrainConfig(
        name="pi05_farm_bottle_lora_gse",
        model=pi0_config.Pi0Config({_MODEL_KWARGS}),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_bottle_lora",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.GSEMergeWeightLoader(
            "{GSE_PARAMS}"
        ),
        freeze_filter=pi0_config.Pi0Config({_MODEL_KWARGS}).get_freeze_filter(),
        ema_decay=None,
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
        print("config.py already patched for bottle-LoRA-off-GSE — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print("ERROR: LeRobotFarmDataConfig not found — run patch_openpi_config.py first.", file=sys.stderr)
        return 1
    wl_path = CFG.parent / "weight_loaders.py"
    if "GSEMergeWeightLoader" not in text and "GSEMergeWeightLoader" not in wl_path.read_text():
        print("ERROR: GSEMergeWeightLoader not found — run patch_openpi_gse_merge.py first.", file=sys.stderr)
        return 1
    anchor = '    TrainConfig(\n        name="pi05_libero"'
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find pi05_libero anchor", file=sys.stderr)
        return 1
    text = text[:idx] + TRAIN_CONFIG_INSERT + text[idx:]
    CFG.write_text(text)
    print(f"patched {CFG}")
    print("  + TrainConfig(name='pi05_farm_bottle_lora_gse')  (π0.5 LoRA off GSE multiobject step-5999)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
