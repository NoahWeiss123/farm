"""Add a π0.5 LoRA fine-tune config (pi05_farm_uf850_lora) to openpi's config.py.

Idempotent. Depends on ``LeRobotFarmDataConfig`` (inserted by
``patch_openpi_config.py`` — run that first; ``setup.sh`` does). Coexists with
both the full-FT ``pi05_farm_uf850`` and the pi0_fast LoRA config; all stay
registered. This patch only *registers* the config — the train sbatch still
trains ``pi05_farm_uf850`` (full FT) by default. To train LoRA instead, set
``CONFIG_NAME=pi05_farm_uf850_lora`` before ``sbatch train_pi05.sbatch``.

WHY a LoRA variant exists
─────────────────────────
The full fine-tune updates all ~3.3B params (PaliGemma 2B + Gemma 300M action
expert) for ~21 epochs on **2 bottle tasks**. On a dataset that small and
narrow, full FT memorizes the two demonstrated trajectories and erodes the
broad visuo-semantic priors π0.5 got from web-scale + cross-embodiment
pretraining — which shows up as (a) replaying canned motions that don't adapt
to the current object pose and (b) complete failure on anything outside the 2
trained prompts.

LoRA freezes the pretrained weights and learns low-rank adapters on top, so the
base capabilities survive the fine-tune. For a 2-task transfer this is the
generalization-preserving recipe (it's exactly what openpi's own ``*_lora``
LIBERO configs do). Bonus: a LoRA fine-tune of π0.5 fits on a **single H100**
(no FSDP needed), which respects the shared cluster's gpu:1 default.

This is NOT a free lunch: LoRA has less capacity to move onto a brand-new
embodiment. If a LoRA run underfits the trained tasks (offline_eval joint MAE
stays high), fall back to full FT but select an EARLIER checkpoint (step-5000 /
10000) — fewer epochs also memorize less. Compare both with
``tools/offline_eval.py`` on held-out frames before committing to one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_UF850_LORA_INSERTED"

# Shared model kwargs — must be identical in ``model=`` and in the
# ``freeze_filter`` instantiation so the freeze filter matches the
# actually-built parameter tree (same pattern as the pi0_fast LoRA config).
_MODEL_KWARGS = (
    "pi05=True, action_horizon=10, discrete_state_input=False, "
    'paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"'
)

TRAIN_CONFIG_INSERT = f'''    #
    # PI05_FARM_UF850_LORA_INSERTED — π0.5 LoRA fine-tune for FARM UF850
    #
    # Generalization-preserving alternative to the full-FT pi05_farm_uf850.
    # Freezes the ~3.3B pretrained backbone and trains low-rank adapters on
    # the LLM + action expert, so π0.5's pretrained priors survive a 2-task
    # transfer. Fits on ONE H100 (no fsdp_devices) — set the sbatch to
    # --gres=gpu:1 when training this. Absolute actions + continuous state +
    # action_horizon=10, identical to the full-FT config so the same
    # serve/eval path (model/eval_pi05.py) works unchanged.
    TrainConfig(
        name="pi05_farm_uf850_lora",
        model=pi0_config.Pi0Config({_MODEL_KWARGS}),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_bottle",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Freeze everything except the LoRA adapters. Same model kwargs as
        # above so the filter lines up with the instantiated tree.
        freeze_filter=pi0_config.Pi0Config({_MODEL_KWARGS}).get_freeze_filter(),
        # EMA off for LoRA (matches openpi's *_lora templates).
        ema_decay=None,
        # batch 32 fits a single 80GB H100. 12k steps ≈ 6.5 epochs — the SAME
        # budget as the GSE config for a head-to-head comparison (~5-7h on 1
        # H100). LoRA's random near-zero init converges slower than GSE's
        # SVD-init head start, so equal-budget is exactly the comparison that
        # surfaces GSE's "faster + better". Checkpoints every 3k for selection.
        batch_size=32,
        num_train_steps=12_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=600,
            peak_lr=1e-4,
            decay_steps=12_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=3_000,
        keep_period=3_000,
        num_workers=16,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for pi05 LoRA — no-op")
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
    print("  + TrainConfig(name='pi05_farm_uf850_lora')  (π0.5 LoRA fine-tune)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
