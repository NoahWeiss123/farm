"""Add a π0.5 GSE config (pi05_farm_multiobject_gse) on the full 424-episode
``NoahWeiss/farm_uf850_multiobject`` dataset (4 tasks: bottle/bear/hat/duck → box).

Idempotent. Depends on ``LeRobotFarmDataConfig`` (patch_openpi_config.py) and the
GSE module + ``GSESVDWeightLoader`` (patch_openpi_gse.py) — setup.sh installs both.
Coexists with the 200-ep ``pi05_farm_uf850_gse``; this is the multi-object,
6-GPU-sized sibling.

Differences from ``pi05_farm_uf850_gse``:
  * ``repo_id="NoahWeiss/farm_uf850_multiobject"`` (424 eps / 129,067 frames, 4 tasks).
  * batch_size=192 = 32/GPU × **6 GPUs** (128 was 32/GPU × 4; 192 is divisible by 6).
  * num_train_steps=6_000 (≈8.9 epochs over 129k frames at batch 192).
  * peak_lr 6e-5 (√-scaled from the 4-GPU config's 5e-5 for the 1.5× larger batch;
    still gentle to protect the SVD-initialized subspace).
  * num_workers=96 (~16/GPU) — pair with --cpus-per-task≈144 so the video-decode
    loader feeds 6 H100s AND the one-time compute_norm_stats pass runs fast.

Train the robust flagship via train_gse_multiobject.sbatch, which exports
FARM_AUG_LEVEL=heavy + FARM_PROMPT_AUG=1 (the heavy-aug + prompt-paraphrase cell).
Same serve/eval path (action_horizon=10, absolute actions) as the other configs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_MULTIOBJECT_GSE_INSERTED"

_GSE_KWARGS = (
    "pi05=True, action_horizon=10, discrete_state_input=False, "
    'paligemma_variant="gemma_2b_gse", action_expert_variant="gemma_300m"'
)

TRAIN_CONFIG_INSERT = f'''    #
    # PI05_FARM_MULTIOBJECT_GSE_INSERTED — π0.5 GSE on farm_uf850_multiobject (424 eps)
    #
    # VLA-GSE (SVD-init adapters on PaliGemma attention, LoRA on FFN, backbone
    # frozen, action expert full FT). Sized for 6× H100 data-parallel (no FSDP).
    TrainConfig(
        name="pi05_farm_multiobject_gse",
        model=pi0_config.Pi0Config({_GSE_KWARGS}),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_multiobject",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.GSESVDWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config({_GSE_KWARGS}).get_freeze_filter(),
        ema_decay=None,
        # 32/GPU × 6 GPUs. 6k steps × 192 = 1.15M samples ≈ 8.9 epochs over 129k frames.
        batch_size=192,
        num_train_steps=6_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=6e-5,
            decay_steps=6_000,
            decay_lr=6e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=1_000,
        keep_period=1_000,
        num_workers=96,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for multiobject GSE — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print("ERROR: LeRobotFarmDataConfig not found — run patch_openpi_config.py first.", file=sys.stderr)
        return 1
    if "GSESVDWeightLoader" not in text and "GSESVDWeightLoader" not in (CFG.parent / "weight_loaders.py").read_text():
        print("ERROR: GSESVDWeightLoader not found — run patch_openpi_gse.py first.", file=sys.stderr)
        return 1
    anchor = '    TrainConfig(\n        name="pi05_libero"'
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find pi05_libero anchor", file=sys.stderr)
        return 1
    text = text[:idx] + TRAIN_CONFIG_INSERT + text[idx:]
    CFG.write_text(text)
    print(f"patched {CFG}")
    print("  + TrainConfig(name='pi05_farm_multiobject_gse')  (π0.5 GSE · 424 eps · 6 GPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
