"""Register ``pi05_farm_multiobject_gse_off_fft`` — a π0.5 GSE e8 fine-tune that
INITIALIZES OFF THE FINISHED 56k FULL FINE-TUNE (``…_fft_robust`` step-55999)
instead of base π0.5.

The question this config answers: once a full fine-tune has converged on the
424-episode multi-object dataset, does stacking GSE spectral-expert capacity on
top of it — frozen FFT backbone + SVD-initialized experts + a few thousand more
gradient steps on the (already-fit) 300M action expert — push the open-loop
joint error DOWN any further, and does it help or hurt the HELD-OUT episodes?

GENUINE HELD-OUT: the FFT init trained on ALL 424 episodes, so a true held-out
signal requires that THIS fine-tune NOT see the held-out episodes. That is
enforced at LAUNCH by train_gse_off_fft.sbatch, which sets FARM_EP_RANGE to the
per-task HEAD episodes only (the eval's TRAIN_RANGES); the per-task TAILS
(HELDOUT_RANGES) are then unseen by the GSE step. The frozen backbone is the same
FFT for both splits, so comparing this model at step-k to its init (FFT@55999)
isolates the 8k GSE continuation's effect — continued fit on heads, generalization
on the never-trained tails. (This config itself names the full dataset; the
episode subset is a data_loader-level env gate, not a config field.)

How it differs from ``pi05_farm_multiobject_gse_sweep`` (the GSE-off-base sweep):
  * ``weight_loader`` source = the local 56k FFT checkpoint, NOT gs://…/pi05_base.
    ``GSESVDWeightLoader`` is unchanged — the FFT is a *full* fine-tune of the
    SAME π0.5 architecture, so its param tree has byte-identical shapes to base;
    only the weight VALUES differ. The loader therefore behaves identically: it
    SVD-initializes the attention GSE adapters off the FFT backbone's spectrum,
    subtracts them from the (now FFT-tuned) frozen ``w``, and ``_merge_params``
    drops the FFT's dense 300M action-expert weights straight in (no adapters →
    they resume from the FFT's fit state, fully trainable).
  * ``num_train_steps=8_000`` with its OWN 8k cosine (``decay_steps=8_000``) so
    the LR schedule actually completes inside the short run. (The sweep's 56k
    decay would leave the LR pinned near peak for the whole 8k — not what we
    want when continuing off an already-converged model.) A short 500-step
    warmup protects the loaded fit weights during the first GSE updates.
  * ``save_interval=keep_period=1_000`` → a DENSE checkpoint ladder
    (1k,2k,…,7k + final 7999) so the error-vs-step curve off the FFT is clean.

Everything else is recipe-matched to the GSE sweep / FFT flagship: batch 32 on
ONE H100 (frozen 2B backbone + adapters + 300M action expert fit at batch 32),
GSE ``gemma_2b_gse`` attention experts + plain ``gemma_300m`` action expert,
``ema_decay=None`` (GSE serves raw params), heavy aug + prompt paraphrase gated
by the sbatch env. Expert count comes from ``FARM_GSE_NUM_SPECIALIZED`` (env,
default 7 = 8 total experts = "e8"), read in ``gemma.get_config`` exactly as the
sweep does — so this config and the sweep share the env-var machinery.

The FFT init params are staged locally by ``train_gse_off_fft.sbatch`` to
``$HOME/farm-train/fft_init/step-55999/params`` (hard-coded below to match;
``GSESVDWeightLoader`` is only invoked at fresh-training init — the eval path
restores from the checkpoint dir and never touches this path).

Idempotent (sentinel-guarded). Depends on ``LeRobotFarmDataConfig``
(patch_openpi_config.py), GSE (patch_openpi_gse.py), and the env patch
(patch_openpi_gse_experts_env.py). Coexists with every other registered config.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FARM_MULTIOBJECT_GSE_OFF_FFT_INSERTED"

# Where train_gse_off_fft.sbatch stages the 56k FFT checkpoint's EMA params.
FFT_INIT_PARAMS = "/home/nhweiss/farm-train/fft_init/step-55999/params"

_GSE_KWARGS = (
    "pi05=True, action_horizon=10, discrete_state_input=False, "
    'paligemma_variant="gemma_2b_gse", action_expert_variant="gemma_300m"'
)

TRAIN_CONFIG_INSERT = f'''    #
    # PI05_FARM_MULTIOBJECT_GSE_OFF_FFT_INSERTED — π0.5 GSE e8 fine-tune initialized
    # OFF the finished 56k FULL fine-tune (…_fft_robust step-55999), not base π0.5.
    #
    # Frozen backbone = the FFT's tuned 2B VLM; GSE attention experts SVD-init off
    # THAT backbone's spectrum; the 300M action expert resumes from the FFT's fit
    # weights and keeps training. 8k steps, dense 1k checkpoints. Tests whether
    # adding GSE capacity on top of a converged full FT lowers joint error further
    # (and whether it helps held-out more than train). Recipe otherwise matched to
    # pi05_farm_multiobject_gse_sweep; ONLY the init source, the 8k cosine, and the
    # 1k checkpoint cadence differ. Expert count from FARM_GSE_NUM_SPECIALIZED
    # (default 7 = 8 total experts = "e8").
    TrainConfig(
        name="pi05_farm_multiobject_gse_off_fft",
        model=pi0_config.Pi0Config({_GSE_KWARGS}),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_multiobject",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        # SVD-initialize the GSE experts off the 56k FFT checkpoint (EMA params)
        # instead of gs://…/pi05_base. Same loader, same shapes (the FFT is a
        # full-FT of the same π0.5 arch) — only the source weights differ. Staged
        # locally by train_gse_off_fft.sbatch. NOT touched on the eval path
        # (create_trained_policy restores from the checkpoint dir).
        weight_loader=weight_loaders.GSESVDWeightLoader(
            "{FFT_INIT_PARAMS}"
        ),
        freeze_filter=pi0_config.Pi0Config({_GSE_KWARGS}).get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        # Short standalone run: 8k steps with its OWN cosine so the schedule
        # completes inside the run (the sweep's 56k decay would sit near-peak the
        # whole time). 500-step warmup protects the loaded fit weights.
        num_train_steps=8_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=8_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Dense checkpoints for a clean error-vs-step curve: every 1k, keep all
        # (1k,2k,…,7k + a final-step save at 7999).
        save_interval=1_000,
        keep_period=1_000,
        num_workers=24,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for GSE-off-FFT — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print("ERROR: LeRobotFarmDataConfig not found — run patch_openpi_config.py first.", file=sys.stderr)
        return 1
    wl_py = (CFG.parent / "weight_loaders.py")
    if "GSESVDWeightLoader" not in text and "GSESVDWeightLoader" not in wl_py.read_text():
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
    print("  + TrainConfig(name='pi05_farm_multiobject_gse_off_fft')  (π0.5 GSE e8 · init off 56k FFT · 1 GPU · 8k)")
    print(f"  weight_loader source: {FFT_INIT_PARAMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
