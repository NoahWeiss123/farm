# Flagship π0.5 GSE-robust on farm_uf850_multiobject — run manifest

**Status:** TRAINING — SLURM job **190** on slinky-0 (6× H100). Dataset published:
[`NoahWeiss/farm_uf850_multiobject`](https://huggingface.co/datasets/NoahWeiss/farm_uf850_multiobject) (public, tag `v2.0`).
Live status in `status.md`; loss curve in `progress.jsonl`; final in `summary.md`.

## Dataset (the "full" set)

- **Source:** all 424 **labeled** episodes of FARM `dataset4`, integrity-gated
  (both cams present, frame counts == meta; 0 excluded). 129,067 frames (~72 min @30fps).
- **4 tasks:** bottle (299), stuffed bear (50), hat (40), rubber duck (35).
- **Publishing to:** `NoahWeiss/farm_uf850_multiobject` (public, LeRobot v2.0, tagged `v2.0`).

## Model

- **Config:** `pi05_farm_multiobject_gse` — VLA-GSE (SVD-init adapters on PaliGemma
  attention + LoRA on FFN, backbone frozen, action expert full FT).
- **Flagship "robust" cell:** `FARM_AUG_LEVEL=heavy` + `FARM_PROMPT_AUG=1`.
- **Checkpoints → `NoahWeiss/farm_uf850_multiobject_gse_robust`** (streamed every 1k steps) + cluster home.

## Resources / hyperparameters

| | |
|---|---|
| GPUs | **6× H100** (data-parallel, no FSDP) — targets idle slinky-0 |
| CPUs / workers | `--cpus-per-task=144` · num_workers=96 (feed 6 GPUs + fast norm-stats) |
| batch_size | 192 (32/GPU) |
| num_train_steps | **6,000** (≈8.9 epochs) |
| lr | cosine, warmup 300, peak 6e-5 → 6e-6 |
| save/keep | every 1,000 steps |

## Data-integrity checks (your "junk in → junk out" requirement)

- [x] Source episodes integrity-gated (cam presence + frame-count match) — 0 bad.
- [x] **Fixed stale prompt-aug** — paraphrase pools were bottle-only; added bear/hat/duck
      (else 3 of 4 tasks would train with no prompt augmentation).
- [x] Verified "heavy" visual aug is sane: bounded photometric on all cams + mild
      crop/rotate (0.90, ±8°) on **base cam only** (wrist grasp view untouched),
      **no flips**, clipped to [0,1]. Train-only; serving unaffected.
- [x] Post-export frame verification — **PASS**: MP4-vs-source PSNR 37–43 dB, frame
      counts match, 640×480, no R/B channel swap, no black/dupe frames; visual
      spot-check of bottle + bear (base & wrist) shows real scenes / natural colors;
      states/actions finite, joints within ±2π, gripper in [0,1].

## Source files

- `model/cluster/patch_openpi_config_pi05_gse_multiobject.py`
- `model/cluster/train_gse_multiobject.sbatch`
- `model/cluster/farm_prompt_aug.py` (updated: 4-task paraphrase pools)

**GPU teardown:** sbatch self-terminates after training → GPUs released; confirmed in `summary.md`.
