"""Add the π0.5 FULL fine-tune config (pi05_farm_multiobject_fft) on the full
424-episode ``NoahWeiss/farm_uf850_multiobject`` dataset (4 tasks:
bottle/bear/hat/duck → box).

This is the **standard π0.5 full fine-tune** pathway — the recipe Physical
Intelligence expect users to run: every parameter trainable (PaliGemma 2B
backbone + Gemma 300M action expert), no ``freeze_filter``, no LoRA/GSE
spectral tricks. It is the full-FT sibling of ``pi05_farm_multiobject_gse``
(the SVD-expert variant) and the multi-object successor to the 2-task
``pi05_farm_uf850`` (which over-memorised and eroded the base). The wager
here: 4 tasks + heavy domain-randomization aug + prompt paraphrase give full
FT enough diversity/regularisation to fit hard *without* collapsing the
pretrained priors the way the 2-task run did — and checkpoint selection picks
the epoch before any over-memorisation sets in.

Idempotent (sentinel-guarded). Depends on ``LeRobotFarmDataConfig``
(patch_openpi_config.py — run first; setup.sh does this). Coexists with every
other registered FARM config. Robustness comes from the env-gated patches
already installed in this openpi tree (setup-time):
  * heavy image domain randomization — ``FARM_AUG_LEVEL=heavy`` (patch_openpi_aug.py):
    wide brightness/contrast/saturation **+ hue** jitter, per-channel gamma
    (colour-temperature), occasional grayscale, gaussian blur, stronger
    crop/rotate. The exact recipe the GSE-robust flagship used.
  * prompt paraphrase — ``FARM_PROMPT_AUG=1`` (patch_openpi_promptaug.py):
    samples a rephrasing per example across all 4 task strings.
Both are no-ops unless the training sbatch exports the vars (train_fft_multiobject.sbatch
does), so the serve/eval path is byte-identical to stock.

── Sizing (4× H100, fsdp_devices=2) ────────────────────────────────────────
A full FT of π0.5 (~3.3 B params) does NOT fit on one 80 GB H100 — Adam state
alone is ~26 GB on top of ~13 GB params + grads. ``fsdp_devices=2`` shards
params + grads + optimizer state across 2 GPUs. With ``--gres=gpu:4`` that is
**2 FSDP groups → 2 data-parallel replicas**. openpi shards the global batch
across ALL 4 devices on the batch axis, so ``batch_size=32`` → **8 samples/GPU**
for activations, with params/grads/Adam sharded /2 — the *byte-identical*
per-GPU footprint of the original 8-GPU/batch-64 full FT that proved it fits
(64/8 = 8/GPU, /2 sharding). This is the validated memory point; we do not
gamble a multi-hour run on an unproven larger batch. (If a smoke test shows
headroom, bump batch_size — but 32 is the safe default; batch_size must be
divisible by the total GPU count, 4.)

── Budget ───────────────────────────────────────────────────────────────────
129,067 frames / batch 32 ≈ 4,033 steps/epoch. 56k steps ≈ 13.9 epochs.
Heavy aug regularises strongly, so the full-FT sweet spot shifts *later* than
the 2-task run's (~6 epochs) — 14 epochs of budget with checkpoint selection
every 8k (≈2 epochs) brackets it and guards against the late over-memorisation
tail. ``save_interval=keep_period=8000`` keeps 6 permanent checkpoints
(8k,16k,24k,32k,40k,48k) plus a final-step save at **step 55999** (train.py
saves at num_train_steps-1; retained via max_to_keep=1, pushed on the drain) —
7 selectable; the background pusher streams each to HF. peak_lr 2.5e-5 is the
established batch-32 value (√2 below the 8-GPU/batch-64 run's 3.5e-5); a long
2k warmup protects the pretrained features during early full-FT updates; cosine
decay to 2.5e-6. EMA 0.999 (full-FT convention; the saved/served params are EMA).
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
SENTINEL = "PI05_FARM_MULTIOBJECT_FFT_INSERTED"

TRAIN_CONFIG_INSERT = '''    #
    # PI05_FARM_MULTIOBJECT_FFT_INSERTED — π0.5 FULL fine-tune on farm_uf850_multiobject (424 eps)
    #
    # Standard full FT (default non-LoRA variants + no freeze_filter ⇒ whole
    # π0.5 model trains). fsdp_devices=2 shards the ~3.3B params across 2 GPUs;
    # the sbatch requests 4 H100s ⇒ 2 FSDP groups → 2 data-parallel replicas.
    # batch_size=32 → 8 samples/GPU (32/4 devices), params/grads/Adam sharded /2
    # — the proven 8-GPU/batch-64 per-GPU footprint. Heavy visual aug + prompt
    # paraphrase come from FARM_AUG_LEVEL=heavy / FARM_PROMPT_AUG=1 exported by
    # the sbatch (no-op otherwise). 56k steps ≈ 13.9 epochs.
    TrainConfig(
        name="pi05_farm_multiobject_fft",
        model=pi0_config.Pi0Config(
            pi05=True,
            # Keep π0.5's 32-dim universal action space (LiberoInputs pads our
            # 7-DoF state/action up, LiberoOutputs slices back to [:, :7]); do
            # NOT hard-code action_dim or the pretrained action head won't load.
            action_horizon=10,
            discrete_state_input=False,
            # No *_lora variants → default gemma_2b + gemma_300m ⇒ full FT.
        ),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_multiobject",
            base_config=DataConfig(prompt_from_task=True),
            # π0.5 trains on ABSOLUTE actions (pi05_libero sets
            # extra_delta_transform=False); our recorded actions are already
            # absolute next-state joint targets, so disable the delta wrap.
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # No freeze_filter ⇒ every parameter trainable (full fine-tune).
        # EMA on (full-FT convention; LoRA/GSE configs set ema_decay=None).
        ema_decay=0.999,
        # Shard the model across 2 GPUs (full FT won't fit on one). 4 GPUs ⇒
        # 2 FSDP groups → 2 data-parallel replicas.
        fsdp_devices=2,
        # 16 samples/replica — the exact per-GPU footprint the 8-GPU/batch-64
        # full FT proved fits. Bump only if a smoke test shows headroom.
        batch_size=32,
        # Feed 4 GPUs (2 DP replicas) without starving them; pair with a wide
        # --cpus-per-task so the base+wrist h264 decode keeps up.
        num_workers=64,
        num_train_steps=56_000,
        # Established batch-32 peak (√2 below the 8-GPU/batch-64 run's 3.5e-5).
        # Long 2k warmup protects the pretrained features during early full-FT
        # updates; cosine decay to 2.5e-6 over the full budget.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=2.5e-5,
            decay_steps=56_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Save every 8k, keep all multiples ⇒ 6 permanent checkpoints
        # (8k,16k,24k,32k,40k,48k) + a final-step save at 55999 = 7 selectable.
        # Keep the pusher --keep-period in sync.
        save_interval=8_000,
        keep_period=8_000,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for multiobject FFT — no-op")
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
    print("  + TrainConfig(name='pi05_farm_multiobject_fft')  (π0.5 FULL FT · 424 eps · 4 GPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
