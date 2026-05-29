"""Add the π0.5 GSE fine-tune config (pi05_farm_uf850_gse) to openpi's config.py.

Idempotent. Depends on ``LeRobotFarmDataConfig`` (from ``patch_openpi_config.py``)
and on ``patch_openpi_gse.py`` having installed the GSE module + variants +
``GSESVDWeightLoader``. ``setup.sh`` runs all three in order. Coexists with the
full-FT and LoRA configs; all stay registered. The train sbatch trains
``pi05_farm_uf850`` by default — train GSE with ``train_pi05_gse.sbatch``.

WHY GSE (vs the other two architectures)
────────────────────────────────────────
Full FT (pi05_farm_uf850) updates all ~3.3B params → on 2 bottle tasks it
memorizes the trajectories and erodes π0.5's pretrained generalization. LoRA
(pi05_farm_uf850_lora) preserves the base but its random near-zero adapters
under-adapt when precise control is needed. GSE
(VLA-GSE, arXiv:2605.06175) splits each VLM weight by its singular spectrum:
the dominant subspace is kept as an SVD-initialized "generalized" adapter (so
the prior is preserved, not discarded like LoRA's random init), and residual
subspaces become "specialized" adapters that adapt control — getting LoRA's
knowledge-preservation AND closer-to-full-FT adaptation. See
``tools/cluster/openpi_gse.py`` for the implementation and
``tools/FINDINGS.md`` for the comparison plan + smoke test.

Config choices: VLM = ``gemma_2b_gse`` (GSE adapters on attention; LoRA on FFN);
action expert = ``gemma_300m`` full fine-tuned (the paper's "action head fully
fine-tuned"); backbone frozen. Like LoRA it fits a single H100.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_UF850_GSE_INSERTED"

TRAIN_CONFIG_INSERT = '''    #
    # PI05_FARM_UF850_GSE_INSERTED — π0.5 GSE fine-tune for FARM UF850
    #
    # Generalized & Specialized Experts (VLA-GSE). SVD-initialized adapters on
    # the PaliGemma attention (dominant spectrum preserved + residual adapted),
    # LoRA on the FFN, full backbone frozen, action expert full fine-tuned.
    # Fits one H100 — use --gres=gpu:1. Same serve/eval path (action_horizon=10,
    # absolute actions) as the other two configs.
    TrainConfig(
        name="pi05_farm_uf850_gse",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_gse",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_bottle",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        # SVD-initializes the GSE adapters from the pi05_base weights, then
        # adjusts the frozen backbone (Eq. 12). Must point at the same base
        # checkpoint the full-FT/LoRA configs load.
        weight_loader=weight_loaders.GSESVDWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Freeze the VLM backbone; train the GSE adapters + action expert.
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_gse",
            action_expert_variant="gemma_300m",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        num_train_steps=10_000,
        # Single LR (3e-5) — a compromise between the paper's decoupled GSE
        # (1e-5) and action-head (1e-4) rates, which openpi's single-schedule
        # optimizer doesn't separate. Gentle enough to adapt the SVD subspace
        # without destabilizing it; decoupling is a documented refinement.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=3e-5,
            decay_steps=10_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=2_000,
        keep_period=2_000,
        num_workers=16,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for pi05 GSE — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print(
            "ERROR: LeRobotFarmDataConfig not found in config.py.\n"
            "       Run patch_openpi_config.py first (setup.sh does this).",
            file=sys.stderr,
        )
        return 1
    if "GSESVDWeightLoader" not in text and "GSESVDWeightLoader" not in (CFG.parent / "weight_loaders.py").read_text():
        print(
            "ERROR: GSESVDWeightLoader not found — run patch_openpi_gse.py first "
            "(setup.sh does this).",
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
    print("  + TrainConfig(name='pi05_farm_uf850_gse')  (π0.5 GSE fine-tune)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
