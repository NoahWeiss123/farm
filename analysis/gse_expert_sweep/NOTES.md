# GSE expert-count sweep — run notes

**Goal.** Find the best number of GSE experts for π0.5 on the multi-object
dataset, and probe whether continued training past the FFT horizon helps. Six
GSE fine-tunes that vary ONLY the expert count, each recipe-matched to the
finished FFT flagship so everything is directly comparable, then benchmark every
checkpoint against training episodes and write up which configuration wins.

## The sweep
GSE (VLA-GSE) splits a frozen weight's singular spectrum into **1 always-on
generalized expert + N specialized experts**. "Number of experts" = total =
`1 + num_specialized`. The default ships at 8 (num_specialized=7). We sweep
½/¼/2×/4×/10× of that:

| run (HF repo suffix) | total experts | num_specialized | adapter rank (2+2·ns) | steps |
|---|---|---|---|---|
| `_gse_e2`  | 2  (¼×)        | 1  | 4   | 56k |
| `_gse_e4`  | 4  (½×)        | 3  | 8   | 56k |
| `_gse_e8`  | 8  (default)   | 7  | 16  | **150k** (long) |
| `_gse_e16` | 16 (2×)        | 15 | 32  | 56k |
| `_gse_e32` | 32 (4×)        | 31 | 64  | 56k |
| `_gse_e80` | 80 (10×)       | 79 | 160 | 56k |

Expert count is selected per-job by the `FARM_GSE_NUM_SPECIALIZED` env var, read
inside `gemma.get_config` — so all six share ONE registered config
(`pi05_farm_multiobject_gse_sweep`) and differ only by that var. The
`GSESVDWeightLoader` derives the count from param shapes, so SVD-init adapts
automatically.

## Recipe (matched to the FFT flagship `pi05_farm_multiobject_fft`)
| knob | value | note |
|---|---|---|
| dataset | `NoahWeiss/farm_uf850_multiobject` | 424 eps / 129,067 frames / 4 tasks |
| method | GSE: frozen 2B backbone + SVD/LoRA adapters + full 300M action expert | vs FFT = every param trainable |
| GPUs | 1× H100 / run | frozen backbone fits one GPU; 6 run in parallel |
| batch | 32 | SAME global batch as the FFT (which sharded 32 across 4 GPUs) |
| steps | 56,000 (≈13.9 ep); e8 → 150,000 | same step budget as FFT |
| checkpoints | 8k,16k,24k,32k,40k,48k + final 55999 | same as FFT (e8 also 56k,64k…144k,149999) |
| LR | cosine warmup 2k, peak 2.5e-5 → 2.5e-6, decay_steps 56k | IDENTICAL to FFT. (optax clamps at floor past 56k → e8's 56k ckpt is recipe-identical; 56k→150k = extended annealed training at the 2.5e-6 floor) |
| EMA | none | GSE convention (FFT used 0.999); served params are the raw params |
| robustness | `FARM_AUG_LEVEL=heavy` + `FARM_PROMPT_AUG=1` | same domain-randomization cell as both flagships |
| CPU | `--cpus-per-task=32`, `num_workers=24` | feed 1 GPU's base+wrist h264 decode |
| partitions | small (56k, ≤24h) / medium (e8 150k) | + resume safety net |

## Benchmark (after training)
`eval_train_endhorizon.py` — open-loop / teacher-forced action prediction on
TRAINING episodes (the standard offline VLA fit metric): for random mid-episode
frames, how well does the predicted 10-step joint chunk's end (`pred[9]`) match
the demonstrated `state[t+10]` (~333 ms ahead). **Fixed seed → identical frames
for every model and checkpoint.** Run per model over its whole checkpoint ladder
(6 GSE + the FFT). Headline = horizon-end joint MAE (°) and within-5° accuracy.
Eval is clean (no aug, canonical prompts).

## Files (`model/cluster/`, deployed to `~/farm-train/`)
- `patch_openpi_gse_experts_env.py` — gemma.py reads FARM_GSE_NUM_SPECIALIZED
- `patch_openpi_config_pi05_gse_sweep.py` — registers `pi05_farm_multiobject_gse_sweep`
- `train_gse_sweep.sbatch` — 1-GPU trainer (env-selected experts, per-run HF repo, resume)
- `smoke_gse_sweep.sbatch` — 79-expert GO/NO-GO (env wiring + OOM + s/step)
- `eval_gse_sweep.sbatch` + `eval_gse_sweep_inner.sh` — per-model ckpt-ladder eval
- `make_gse_expert_report.py` — figures + report (runs locally over `raw/`)

## Reference
- FFT flagship: `NoahWeiss/farm_uf850_multiobject_fft_robust` — DONE, ckpts
  step-8000…step-55999 on HF. Train loss 0.078→~0.0007 over 56k (job 406).

## Status log
<!-- appended live -->
- _setup_ (2026-06-02): wrote + deployed all scripts. Applied env-var patch
  (2 GSE sites in gemma.py) + sweep config patch (registered, compiles).
  Verified LR clamps at the 2.5e-6 floor past step 56k (optax). Staged norm
  stats into the new config's assets (skips the 24-min recompute, avoids a
  6-way write race). Submitted 79-expert smoke (job 474). Launched a 6-reviewer
  adversarial pre-launch review.
- _review_ → GO-WITH-FIXES (no blockers). Applied: (1) train default partition
  → `medium` / 2-day (batch-32-on-1-GPU ≈ GSE flagship's 1.4-1.7 s/step ⇒ 56k
  would straddle small's 24h cap) + tag↔count↔repo sanity-map guard; (2) eval
  exports FARM_GSE_NUM_SPECIALIZED only when non-empty (empty ⇒ int("") crash);
  (3) eval-time env↔ckpt wiring assert; (4) report "best ckpt" restricted to
  ≤56k so the 150k run can't cherry-pick + EMA(FFT)-vs-raw(GSE) note. Redeployed.
- _smoke_ (job 474, COMPLETED exit 0): **SMOKE_GSE_OK**. E=80 (num_specialized=79,
  rank 160) fits batch 32 on 1 GPU — **no OOM** (binding constraint cleared ⇒ all
  smaller configs fit). ENV_WIRING_OK (env→num_specialized verified). Step0
  loss=0.078, Step100 loss=0.059 (healthy, ≈ FFT/GSE flagships). Norm stats reused
  (no recompute). train.py 748s/120 steps incl one-time 80-expert JIT compile.
- _launch_ (2026-06-02): 6 runs on `medium`, 1 GPU each:
  | job | tag | num_specialized | total exp | steps | walltime | HF repo |
  |---|---|---|---|---|---|---|
  | 479 | e2  | 1  | 2  | 56k  | 3d | `…_gse_e2`  |
  | 480 | e4  | 3  | 4  | 56k  | 3d | `…_gse_e4`  |
  | 481 | e8  | 7  | 8  | 150k | 5d | `…_gse_e8`  |
  | 482 | e16 | 15 | 16 | 56k  | 3d | `…_gse_e16` |
  | 483 | e32 | 31 | 32 | 56k  | 3d | `…_gse_e32` |
  | 484 | e80 | 79 | 80 | 56k  | 3d | `…_gse_e80` |
  EXP_NAME = `farm_gse_sweep_<tag>_<jobid>`; ckpts → `checkpoints/pi05_farm_multiobject_gse_sweep/<EXP_NAME>/`.

## ⚠ Cross-session resource conflict (2026-06-02 ~09:55Z)
A SECOND session on the same `nhweiss` account (a parallel **fftlora** training
sweep — `train_fftlora.sbatch`, 6 task variants, 4h each, launched ~09:43–09:56
from a different exec on the one login pod) cleared the account's queue when it
launched: it **`scancel`'d my running e2/e4 (479/480)** and **held e8/e16/e32/e80
(481–484)** to claim GPUs. The cluster is NOT scarce (~15/32 GPUs used; slinky-1
mostly idle), and its `monitor_fftlora.sh` is read-only (no recurring killer) —
so this was a one-time launch-time clear, not a persistent war. Resolution: left
fftlora untouched, **released 481–484 + resubmitted e2/e4 as 501/502**; both
workstreams have ample capacity to coexist. Watching for any re-clear; will
escalate to the user only if a persistent conflict re-emerges. **Updated job IDs:
e2=501, e4=502, e8=481, e16=482, e32=483, e80=484.**

## ⏸ PAUSED by user (2026-06-02 ~18:48Z)
User asked to stop and revisit later. Cancelled all 6 GSE jobs (481/482/483/484/
501/502/516/517 → CANCELLED); GPUs released; persistent monitor stopped. The
parallel fftlora session was left untouched. **Infrastructure is preserved for a
fast resume** — env-var patch + sweep config applied & compiling, all sbatch/eval/
report scripts deployed to `~/farm-train/`, norm stats staged, HF repos created
(e32 already has a real `step-8000` checkpoint on HF).

**Decisions locked for resume** (from the 8h-limit discussion):
- Five 56k runs → re-scope to **num_train_steps=8000, save_interval=1000** (dense
  curve), **8h walltime**, 1 GPU each (8h on 1 GPU ≈ 8k GSE steps; the FFT only hit
  56k in 8h because it used 4 GPUs). Compare across steps 1k→8k + FFT@8k.
- **e8 exempt** — keep the full 150k continued-training run.
- To resume: edit the sweep config (num_train_steps 56000→8000, save/keep 8000→1000),
  redeploy, and submit `train_gse_sweep.sbatch` per run with `-t 8:00:00` (e8 with
  `-t 5-00:00:00` + NUM_TRAIN_STEPS=150000). Watch the per-user ~6-GPU cap shared
  with the fftlora session.

## Live training log (≈10-min cadence)
| wall | e2(501) | e4(502) | e8(481) | e16(482) | e32(483) | e80(484) | notes |
|---|---|---|---|---|---|---|---|
| 09:41 launch | R→cancelled | R→cancelled | held | held | held | held | cross-session clear |
| 09:57 relaunch | PD | PD | PD(rel) | PD(rel) | PD(rel) | PD(rel) | all eligible, fftlora left alone |
| 10:10 | PD | PD | PD | PD | PD | PD | **per-user GPU cap (~6) hit by the 6 fftlora jobs** ⇒ my gse jobs queued, NOT killed. `squeue --start`: e8 ETA 13:56 (= fftlora 4h walltime end). Self-resolving: gse starts as fftlora frees the cap. No escalation — normal queue wait. |
| 10:10–13:56 | PD | PD | PD | PD | PD | PD | ~3.8h queue wait while fftlora ran (jobs healthy, never cancelled again). |
| 13:56 | PD | PD | **R** | **R** | PD | PD | fftlora hit its 4h cap → freed the quota. e8(481) + e16(482) started first; e32/e80/e2/e4 queued for GPUs. |
| 14:45 | PD(516) | PD(517) | — | — | R Step900 | R Step500 | **coexistence stable** — other session's eval/extract just queue (no longer cancelling mine). e32≈1.6 s/step, e80≈3.4 s/step (matches smoke) ⇒ e80/56k≈53h, e8/150k≈60-90h. Other 4 pending on genuine cluster-wide GPU availability (not the cap/conflict). Switched to a **persistent event-driven monitor** (wakes on job start/complete/cancel; tolerates Teleport DNS blips) instead of 10-min heartbeats. First checkpoints (step 8000) ≈17:40Z (e32) / ≈21:40Z (e80). |
| 14:08 | PD | PD | **CANCELLED** | **R→CANCELLED** | R | R | **conflict recurred**: other session relaunched fftlora + launched eval-fft/extract and **cancelled my running e8/e16** (it kills running jobs, not just holds pending). Released+restarted: e32(483)/e80(484) now RUNNING; e8 re-queued as **516**, e16 as **517**, e2=501, e4=502. Other session now in eval/extract (wrap-up?). Drain-vs-ramp watch: if it cancels e32/e80 again → escalate; else my jobs fill in as it drains. **Updated IDs: e8=516, e16=517, e32=483, e80=484, e2=501, e4=502.** Nothing of mine checkpointed yet (all cancelled <step 8000), so no data lost. |
