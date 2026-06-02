"""Register ``pi05_farm_multiobject_gse_sweep`` — the π0.5 GSE config used for the
EXPERT-COUNT SWEEP on ``NoahWeiss/farm_uf850_multiobject`` (424 eps, 4 tasks),
sized for a SINGLE H100 so 6 expert-count variants run in parallel (1 GPU each).

This is the GSE sibling of ``pi05_farm_multiobject_fft`` and is deliberately
recipe-MATCHED to it so GSE-vs-FFT and across-expert comparisons are clean at
identical step counts. The ONLY method difference vs the FFT is GSE SVD-spectral
experts on a frozen backbone instead of a full fine-tune.

Matched to the FFT flagship:
  * ``batch_size=32`` — same GLOBAL batch as the FFT (which sharded 32 across 4
    GPUs via fsdp_devices=2). Here it is 32 on ONE GPU: GSE freezes the ~2B VLM
    backbone (only the SVD/LoRA adapters + the 300M action expert train), so it
    fits one 80GB H100. 32/GPU is the proven per-GPU footprint of the 6-GPU /
    batch-192 GSE flagship (``pi05_farm_multiobject_gse``).
  * ``num_train_steps=56_000`` — same as FFT (≈13.9 epochs over 129k frames).
  * ``save_interval=keep_period=8_000`` — same checkpoints as FFT:
    8k,16k,24k,32k,40k,48k + a final-step save at 55999 (train.py saves at
    num_train_steps-1; max_to_keep=1 retains it). 7 selectable per run.
  * LR cosine, ``warmup_steps=2_000``, ``peak_lr=2.5e-5`` → ``decay_lr=2.5e-6``,
    ``decay_steps=56_000`` — IDENTICAL to the FFT. (2.5e-5 is also exactly what
    √-batch-scaling from the GSE flagship's 6e-5@192 gives at batch 32, so it is
    consistent with both the FFT and the GSE lineages.)
  * ``optimizer=AdamW(clip_gradient_norm=1.0)``.
  * No ``fsdp_devices`` — the frozen backbone fits one GPU; single replica.

GSE-specific:
  * ``ema_decay=None`` (GSE convention; the LoRA/GSE serve path expects the raw,
    non-EMA params — only full-FT uses EMA).
  * ``num_workers=24`` — feeds one GPU's base+wrist h264 decode without starving
    it (pair with ``--cpus-per-task≈32``).

EXPERT COUNT is NOT set here. It comes from the ``FARM_GSE_NUM_SPECIALIZED`` env
var read inside ``gemma.get_config`` (see ``patch_openpi_gse_experts_env.py``),
so all 6 sweep jobs share THIS one config name and differ only by that env var:
  total experts  2   4   8(default)  16  32  80
  num_specialized 1   3   7           15  31  79
  adapter rank    4   8   16          32  64  160   (= 2 + num_specialized·2)

The default-expert (8-total / num_specialized=7) job additionally runs LONG:
submitted with ``NUM_TRAIN_STEPS=150000`` (CLI override). ``decay_steps`` stays
56_000, so its step-56000 checkpoint is recipe-identical to the other five and
steps 56k→150k are extended annealed training at the 2.5e-6 LR floor — a clean
test of whether more gradient steps on the same data keep reducing error.

Idempotent (sentinel-guarded). Depends on ``LeRobotFarmDataConfig``
(patch_openpi_config.py), GSE (patch_openpi_gse.py), and the env patch
(patch_openpi_gse_experts_env.py). Coexists with ``pi05_farm_multiobject_gse``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_MULTIOBJECT_GSE_SWEEP_INSERTED"

_GSE_KWARGS = (
    "pi05=True, action_horizon=10, discrete_state_input=False, "
    'paligemma_variant="gemma_2b_gse", action_expert_variant="gemma_300m"'
)

TRAIN_CONFIG_INSERT = f'''    #
    # PI05_FARM_MULTIOBJECT_GSE_SWEEP_INSERTED — π0.5 GSE expert-count sweep (1 GPU)
    #
    # Recipe-matched to pi05_farm_multiobject_fft (same batch/steps/ckpts/LR) so
    # GSE-vs-FFT and across-expert comparisons are clean. num_specialized comes
    # from FARM_GSE_NUM_SPECIALIZED (env), default 7. Single H100: frozen 2B
    # backbone + SVD/LoRA adapters + 300M action expert all fit at batch 32.
    TrainConfig(
        name="pi05_farm_multiobject_gse_sweep",
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
        # Same GLOBAL batch as the FFT (32), here on one GPU (frozen backbone).
        batch_size=32,
        # Same step budget as the FFT (≈13.9 epochs). The default-expert run
        # overrides this to 150000 via --num-train-steps; decay_steps stays 56k.
        num_train_steps=56_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2.5e-5,
            decay_steps=56_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Same checkpoints as the FFT: 8k,16k,24k,32k,40k,48k + final 55999.
        save_interval=8_000,
        keep_period=8_000,
        num_workers=24,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for GSE sweep — no-op")
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
    print("  + TrainConfig(name='pi05_farm_multiobject_gse_sweep')  (π0.5 GSE · 424 eps · 1 GPU · expert sweep)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
