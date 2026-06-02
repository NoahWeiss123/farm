# Full Fine-Tune (FFT) flagship on the multi-object dataset — run notes

**Goal.** Train a *super-good* π0.5 **full fine-tune** — the standard Physical
Intelligence full-FT pathway (every param trainable; no LoRA, no GSE spectral
experts) — on the 4-task multi-object dataset, made robust with the same
domain-randomization toolkit built for the GSE flagship. Then benchmark it
in-depth against the other techniques (GSE-robust, LoRA, 2-task full FT).

This is the FFT counterpart to the GSE flagship
(`NoahWeiss/farm_uf850_multiobject_gse_robust`) on the *same* dataset, so the
two are directly comparable.

## Dataset
- `NoahWeiss/farm_uf850_multiobject` — 424 episodes / 129,067 frames / 4 tasks
  (bottle / bear / hat / duck → box). Same set the GSE flagship trained on.

## Recipe (the standard full-FT pathway)
| Knob | Value | Why |
|---|---|---|
| Model | π0.5, `pi05=True`, full FT | default gemma_2b + gemma_300m, **no** freeze_filter → every param trains. The pathway PI expect. |
| Data | `LeRobotFarmDataConfig`, absolute actions | `use_delta_joint_actions=False` (π0.5 convention) |
| GPUs | 4× H100, `fsdp_devices=2` | full FT (~3.3B) won't fit one 80GB GPU; shard /2 → 2 FSDP groups → **2 data-parallel replicas** |
| Batch | 32 (→ 8 samples/GPU) | batch sharded across all 4 devices → 8/GPU activations, params/grads/Adam /2 — *byte-identical* to the proven 8-GPU/batch-64 footprint. (smoke probes batch 48 for headroom) |
| Steps | 56,000 (≈13.9 epochs) | heavy aug regularises → sweet spot shifts later than the 2-task run's ~6 ep; checkpoint-select every 8k |
| LR | cosine, warmup 2k, peak **2.5e-5** → 2.5e-6 | established batch-32 peak; long warmup protects pretrained features during early full-FT updates |
| EMA | 0.999 | full-FT convention (saved/served params are EMA) |
| Robustness | `FARM_AUG_LEVEL=heavy` + `FARM_PROMPT_AUG=1` | reuse the GSE-era domain randomization: wide brightness/contrast/saturation **+ hue** jitter, per-channel gamma (colour-temp), grayscale, gaussian blur, crop/rotate; + prompt paraphrase across all 4 tasks |
| Checkpoints | 8k,16k,24k,32k,40k,48k + final **55999** = 7 | streamed to HF `NoahWeiss/farm_uf850_multiobject_fft_robust`, tagged `step-N` (eval pins `step-55999`, not 56000) |
| CPU | `--cpus-per-task=128`, `num_workers=64` | 2 CPU/worker (matches the proven full-FT run) so base+wrist h264 decode never starves the GPUs |
| Walltime | `--time=16:00:00` | ~9–10 h expected; margin for dataloader stalls so the final/best checkpoint always saves |

## Why these choices (vs the failed 2-task full FT)
The deployed 2-task full FT (`pi05_farm_uf850`, step-19999) memorised the two
near-inverse bottle trajectories and eroded π0.5's pretrained priors → ~30 %
on trained tasks, total OOD failure, jitter (see `model/FINDINGS.md`). This run
attacks both failure modes *within the full-FT pathway*:
- **More + more diverse data** (4 tasks, 424 eps vs 2 tasks, 200 eps) → harder
  to memorise a single trajectory per prompt.
- **Heavy visual domain randomization** → the encoder can't overfit one room's
  appearance (the real live-failure cause was a room/camera domain shift).
- **Prompt paraphrase** → keeps the language pathway alive (the 2-task tasks
  aliased at their endpoints; only the prompt disambiguates).
- **Checkpoint selection** every ~2 epochs → pick the epoch before the
  over-memorisation tail, not the last step.

## Files
- `model/cluster/patch_openpi_config_pi05_fft_multiobject.py` — registers `pi05_farm_multiobject_fft`
- `model/cluster/train_fft_multiobject.sbatch` — the 4-GPU job
- `model/cluster/smoke_fft_multiobject.sbatch` — 5-step 4-GPU GO/NO-GO smoke

## Status log
<!-- appended live during the run -->
- _setup_: wrote config + sbatch + smoke; ran a 5-reviewer adversarial pre-launch
  review (FSDP/memory, config-vs-API, aug wiring, sbatch, disk/budget). Verdict
  GO_WITH_FIXES. Applied all fixes:
  - **resume blocker**: `EXP_NAME` now derives from `RESUME_FROM` when set (was
    embedding the job id → empty-dir restart-from-0). Resume = resubmit with
    `RESUME_FROM=<old EXP_NAME>`.
  - **walltime**: `--time` 12→16 h; `--cpus-per-task` 112→128 (2 CPU/worker).
  - **disk**: free only `farm_uf850_pi05_113` (167 GB, *confirmed* on HF). Keep
    `_82` (63 GB) — it is **not** backed up anywhere.
  - **smoke**: now `rm`s its ~84 GB orphan checkpoints; keeps the norm-stats the
    real run reuses.
  - **docs**: final checkpoint is step **55999** (not 56000); per-GPU = 8 samples.
- _smoke (job 400)_: 4-GPU 5-step GO/NO-GO. Norm-stats pass over 129k frames took
  ~27 min, then **batch-32 validated**: `SMOKE_FFT_OK`, Step 0 `loss=0.0776
  grad_norm=0.92 param_norm=1802` (finite, healthy, ≈ GSE run's 0.0742), **no OOM**
  on the real 4-GPU FSDP layout. The batch-48 headroom probe was inconclusive (the
  45-min smoke walltime expired mid-probe) — irrelevant, we launch at the proven
  batch 32. Norm stats now cached → the real run skips that 27-min pass.
- _launch_: **job 406** `farm-fft-multiobj`, exp `farm_fft_multiobject_robust_406`,
  4× H100, batch 32, 56k steps, heavy aug + prompt aug. Checkpoints stream to
  `NoahWeiss/farm_uf850_multiobject_fft_robust`. Disk at launch 477 GB (peak ~770).
  10-min progress monitor running (`monitor_fft.sh 406`).

## Live training log (10-min cadence)
| wall | step / 56k | loss | grad_norm | s/step | ETA | checkpoints (HF) |
|---|---|---|---|---|---|---|
| 00:31 | building | — | — | — | — | — |
| ~00:34 | startup: norm-stats recompute (24 min) + JIT compile | — | — | — | — | — |
| 00:40 | 500 | 0.0136 | 0.228 | 0.49 | ~7h33m | — |
| 02:00 | 10000 | 0.0029 | — | 0.50 | — | 8k |
| 04:03 | 24000 | 0.0019 | — | 0.50 | — | 8k,16k,24k |
| 06:13 | 40000 | 0.0011 | — | 0.50 | — | …40k |
| 08:31 | **55999 (done)** | **~0.0007** | 0.030 | 0.50 | — | 8k,16k,24k,32k,40k,48k,55999 |

**FFT-56k COMPLETE** (job 406, `COMPLETED`, 8h31m wall, exit 0). All 7 checkpoints
on HF `NoahWeiss/farm_uf850_multiobject_fft_robust` (`step-8000`…`step-55999`).
Train loss fell monotonically 0.078 → ~0.0007 over 56k steps — a clean fit, but the
monotonic descent on a fixed dataset is exactly why we do **checkpoint selection**
on held-out data rather than assume the 55999 final is best.

Next: checkpoint-generalization sweeps (jobs 470 training-data + 471 held-out
eval_bench) → select best step → that becomes both the reported FFT and the LoRA base.

Startup note: the real run **recomputed** norm-stats (~24 min) despite the smoke
having cached them — this openpi's `compute_norm_stats` recomputes unconditionally
rather than skip-if-present. ~3 % of the 16 h budget; harmless. At 0.49 s/step the
run is faster than the reviewer's 0.56 projection → ~8 h total incl. startup.

## Phase C — analysis plan (infra staged, runs after training)
`eval_fft_bench.sbatch` (1 GPU) will, on the held-out **eval_bench** (15 eps / 5 tasks):
1. **Sweep every FFT checkpoint** (8k…55999) clean + 10 domain-shift conditions →
   a training/robustness curve and checkpoint selection.
2. Eval the comparison techniques on the **same** bench, same openpi build:
   - **GSE-multiobject-robust** — the controlled foil (same 424-ep/4-task data +
     same heavy-aug + prompt-aug; differs only full-FT vs SVD-spectral-experts).
   - **LoRA** (100-ep bottle) and the **2-task full FT** (the over-memorised baseline).
3. `make_fft_report.py` → head-to-head figures + FINDINGS, written **data-driven**
   after the eval. Headline: does full FT, with the GSE robustness toolkit and 2×
   the data, match or beat GSE on this multi-object set — clean fit AND under shift?
- Interim: when step-8000 lands (~1 h) I'll run a quick `fftonly` eval to validate
  the pipeline end-to-end before the full sweep.
