"""Add ``pi05_fftlora`` — a π0.5 LoRA that initializes off the **full-FT
multi-object checkpoint** (NoahWeiss/farm_uf850_multiobject_fft_robust, step-55999)
instead of base π0.5. ONE config, reused for every per-task LoRA: the task is
selected at runtime via ``FARM_EP_RANGE`` (episode subset of the multi-object
dataset), so all task-LoRAs share an identical config + identical norm stats —
exactly the controlled setup the LoRA-vector ("skills") analysis needs.

Design (why each choice):
  * weight_loader = CheckpointWeightLoader(<FFT-56k params>). The FFT checkpoint is
    a DENSE full fine-tune of pi05_base, so its param tree has the same keys as
    pi05_base; loading it into the gemma_2b_lora model populates the (now frozen)
    dense base and leaves lora_a/lora_b at init — i.e. the LoRA is a pure low-rank
    delta ON TOP of the FFT-56k policy. (Same mechanism as pi05_farm_bottle_lora's
    CheckpointWeightLoader(pi05_base); only the dense source differs. The
    *_gse sibling needed GSEMergeWeightLoader to collapse adapters first — the
    FFT base has none, so the plain loader suffices.)
  * repo_id = the FULL multi-object dataset; the per-task episode subset is chosen
    at runtime with FARM_EP_RANGE (bottle 0:299, bear 299:349, duck 349:384,
    hat 384:424; equal-size variants use the first N of each task's range).
  * SHARED norm stats: the LoRA sbatch pre-places the FFT base model's
    full-dataset norm_stats.json into this config's asset dir, so every task-LoRA
    normalizes inputs identically to the FFT base → the LoRA delta is the ONLY
    variable across tasks (clean vector comparison).
  * NO augmentation at train time (the sbatch exports FARM_AUG_LEVEL=off and
    leaves FARM_PROMPT_AUG unset) — each task sees one fixed prompt + unshifted
    frames, per the experiment spec.
  * Identical hyperparameters for every task-LoRA (same rank, steps, LR, seed) so
    the resulting adapters are directly comparable as vectors.

Idempotent. Depends on LeRobotFarmDataConfig (patch_openpi_config.py). The FFT-56k
checkpoint must exist before training (set after the full-FT run finishes). The
init params path is overridable via FFT_INIT_PARAMS.

    python3 patch_openpi_config_pi05_fftlora.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "PI05_FFTLORA_INSERTED"

# FFT checkpoint to LoRA off of. NOT the final 56k step blindly — the BEST-
# generalizing FFT checkpoint (selected post-training by the eval_fft_bench fftonly
# sweep, to guard against the final step over-memorising). We point at a stable
# `lora_base` symlink that is set to the chosen step dir after selection; the LoRA
# never has to be re-registered when the choice changes.
FFT_PARAMS = os.environ.get(
    "FFT_INIT_PARAMS",
    "/home/nhweiss/farm-train/openpi/checkpoints/pi05_farm_multiobject_fft/"
    "farm_fft_multiobject_robust_406/lora_base/params",
)

# LoRA adapters on the frozen (FFT-initialized) backbone + action expert — IDENTICAL
# to pi05_farm_bottle_lora so the only differences vs that run are (a) the dense
# init source and (b) the per-task episode subset.
_MODEL_KWARGS = (
    "pi05=True, action_horizon=10, discrete_state_input=False, "
    'paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"'
)

TRAIN_CONFIG_INSERT = f'''    #
    # PI05_FFTLORA_INSERTED — π0.5 LoRA initialized off the full-FT multi-object
    # checkpoint (step-55999). ONE config for ALL per-task LoRAs; the task is the
    # FARM_EP_RANGE episode subset chosen at runtime. Shared norm stats (pre-placed
    # by the sbatch). NO train-time aug. 1× H100 — --gres=gpu:1 --cpus-per-task=64.
    TrainConfig(
        name="pi05_fftlora",
        model=pi0_config.Pi0Config({_MODEL_KWARGS}),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_multiobject",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "{FFT_PARAMS}"
        ),
        # Freeze EVERYTHING except the LoRA adapters. openpi's DEFAULT lora
        # freeze_filter (All(.*llm.*, Not(.*lora.*))) freezes only the LLM dense
        # weights and leaves the ~400M SigLIP image tower + the action proj/MLP
        # heads TRAINABLE — so a per-task LoRA would also fine-tune a separate
        # vision tower + heads, and the "skill" would NOT be a pure low-rank delta.
        # Not(.*(lora|gse).*) freezes all non-adapter params (over the whole model,
        # not just llm) → the ONLY trainable params are lora_a/lora_b. The skill IS
        # the LoRA matrix on top of the shared frozen FFT base. (nnx/nnx_utils are
        # reached through the pi0_config module so config.py needs no new imports.)
        freeze_filter=pi0_config.nnx.Not(pi0_config.nnx_utils.PathRegex(".*(lora|gse).*")),
        ema_decay=None,
        # Identical protocol for every task-LoRA → directly comparable adapters.
        # batch 32 fits one 80GB H100. 12k steps is "extensive" for a single-task
        # subset (≥12 epochs even on the 100-ep bottle set; ~50 on the 30-ep sets).
        # Checkpoints every 3k → step-3000,6000,9000 + a final-step save at 11999.
        batch_size=32,
        num_train_steps=12_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=12_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=3_000,
        keep_period=3_000,
        num_workers=32,
        # Deterministic init so lora_a/lora_b start IDENTICAL across every task →
        # the trained adapter difference reflects the TASK, in a shared coordinate
        # frame. The shared-init does add a task-independent A0·B0 baseline to the
        # pairwise cosine; we MEASURE it with a same-task/different-seed control
        # (bottle30 @ seed 1, via --seed) and report cross-task similarity relative
        # to that ceiling. Override per run with --seed.
        seed=42,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched for pi05_fftlora — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print("ERROR: LeRobotFarmDataConfig not found — run patch_openpi_config.py first.", file=sys.stderr)
        return 1
    anchor = '    TrainConfig(\n        name="pi05_libero"'
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find pi05_libero anchor", file=sys.stderr)
        return 1
    text = text[:idx] + TRAIN_CONFIG_INSERT + text[idx:]
    CFG.write_text(text)
    print(f"patched {CFG}")
    print(f"  + TrainConfig(name='pi05_fftlora')  (LoRA off FFT-56k: {FFT_PARAMS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
