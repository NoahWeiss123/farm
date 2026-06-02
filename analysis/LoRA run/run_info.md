# π0.5 LoRA on farm_bottle_lora — run manifest

**SLURM job:** `188`  ·  **node:** slinky-2  ·  **submitted:** 2026-05-30
**partition:** small  ·  **resources:** 1× H100 80GB, `--cpus-per-task=64`, `--time=08:00:00`

> **Attempt 1 (job 187) FAILED** at `compute_norm_stats` — the HF dataset repo
> `farm_bottle_lora` was missing the LeRobot codebase-version git tag (`v2.0`),
> so the loader raised `RevisionNotFoundError`. Fixed by tagging the repo `v2.0`
> (matching dataset3) + clearing the stale cache, then resubmitting as job 188.
> Diagnosis preserved in `attempt1_failed/`.

## What's training

- **Config:** `pi05_farm_bottle_lora` (π0.5 LoRA — frozen ~3.3B backbone + low-rank
  adapters on the LLM & action expert; absolute actions, continuous state,
  `action_horizon=10`).
- **Dataset:** [`NoahWeiss/farm_bottle_lora`](https://huggingface.co/datasets/NoahWeiss/farm_bottle_lora)
  @ tag `v2.0` — 100 episodes / 26,378 frames, 1 task: *"Picking up the bottle and
  placing it on the box"* (the first 100 of FARM dataset4).
- **Base weights:** `gs://openpi-assets/checkpoints/pi05_base`.

## Hyperparameters

| | |
|---|---|
| batch_size | 32 |
| num_train_steps | 10,000 (≈12 epochs) |
| num_workers | 32 (with 64 CPUs — anti-starvation) |
| lr schedule | cosine, warmup 500, peak 1e-4 → 1e-5 |
| optimizer | AdamW, clip_grad_norm 1.0 |
| ema | off (LoRA) |
| save/keep interval | every 2,000 steps |

## Outputs

- **Checkpoints (cluster, persistent home — NOT pushed to HF):**
  `~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora/pi05_farm_bottle_lora_188/<step>/`
  (step-2000 … step-10000; select by held-out eval, not the last step).
- **This folder** (`analysis/LoRA run/`) is updated live by the monitor:
  - `status.md` — latest human-readable snapshot (state, step, loss, checkpoints)
  - `progress.log` — one timestamped line per poll
  - `progress.jsonl` — parsed `{t, step, loss, state, ckpts}` per poll (loss curve)
  - `train_tail.log` — rolling tail of the SLURM training log
  - `summary.md` — written when the job finishes (final state + GPU released)
  - `attempt1_failed/` — artifacts from the failed first attempt (job 187)

## Source

- `model/cluster/patch_openpi_config_pi05_lora_bottle100.py` (registers the config)
- `model/cluster/train_pi05_lora_bottle100.sbatch` (the job)

**GPU teardown:** the sbatch self-terminates after training → SLURM releases the
GPU automatically. The monitor confirms this in `summary.md`.
