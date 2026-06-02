# FFT-LoRA "skills" program — execution plan & methodology

**Thesis.** On top of a single full-FT π0.5 base (the FFT-56k multi-object model),
train one LoRA per object task — bottle, bear, duck, hat — under an *identical*
protocol (same rank, steps, LR, seed; only the task data differs). Each LoRA is a
low-rank weight delta ΔW = A·B. Ask: (1) does adding a task-LoRA to the base improve
that task on held-out episodes? (2) treated as vectors, do similar tasks yield
similar LoRAs — i.e. is there a usable "skill space" (the hat-vector analogy from
image embeddings)? This would make LoRAs a composable **skills library** over a
frozen generalist.

## Base model
`NoahWeiss/farm_uf850_multiobject_fft_robust` step-55999 ("the 56k model"), trained
this session. LoRA inits off it via `CheckpointWeightLoader(.../55999/params)` +
`gemma_2b_lora`/`gemma_300m_lora` → frozen FFT base + fresh adapters (verified
against openpi `_merge_params(..., missing_regex=".*lora.*")`).

## Dataset (contiguous by task)
bottle 0–298 (299) · bear 299–348 (50) · duck 349–383 (35) · hat 384–423 (40).

## The LoRA set (all: pi05_fftlora config, FARM_AUG_LEVEL=off, no prompt aug, seed 42
## unless noted, batch 32, 12k steps, peak LR 1e-4, **only LoRA adapters trainable**
## (config freezes ALL non-lora incl. the SigLIP vision tower + heads), SHARED norm
## stats = the FFT base's full-set stats)
| LoRA | train eps | seed | role |
|---|---|---|---|
| bottle30 | 0:30 | 42 | **vector set** + scaling vs bottle100 |
| bear30 | 299:329 | 42 | **vector set** (held-out bear 329:349, 20 eps) |
| duck30 | 349:379 | 42 | **vector set** (held-out duck 379:384, 5 eps) |
| hat30 | 384:414 | 42 | **vector set** (held-out hat 414:424, 10 eps) |
| bottle100 | 0:100 | 42 | **primary held-out eval** (bottle 100:299, 199 eps) + data-scaling |
| bottle30s1 | 0:30 | **1** | **control**: same-task/diff-seed cosine ceiling (the shared-init baseline) |

The **equal-size n=30 seed-42 set {bottle30, bear30, duck30, hat30}** is the
controlled vector comparison (same #episodes, #steps, seed → only task differs), and
n=30 leaves EVERY task a genuine held-out tail for the eval. Controls: bottle30s1
(seed) bounds how much of the cosine is shared-init vs task; bottle100 vs bottle30
(data) tests task-vs-data-amount.

**Why only adapters train** (review fix): openpi's default LoRA freeze leaves the
~400M SigLIP vision tower + proj/MLP heads trainable, so the per-task delta would not
be a pure LoRA. We freeze all non-lora params → the skill IS the low-rank LoRA matrix.

Every LoRA streams to its own HF repo `NoahWeiss/farm_fftlora_<task>` (per-task
pusher; final-checkpoint drain). **All models saved to HF** (user requirement).

## Review fixes applied (5-reviewer pre-run audit; 3 blockers + 9 majors)
- **Episode subset**: lerobot 0.1.0 `episodes=` doesn't re-index → IndexError for
  non-zero-start tasks (3/4 LoRAs). Fixed: full dataset + frame-index `torch.Subset`
  (validated on bear 299:329 = 8238 frames).
- **Freeze all-but-LoRA** (above) so the skill is a pure LoRA delta.
- **Tower split**: action expert is the `_1` suffix, not substring 'action' — fixed +
  asserts both towers non-empty.
- **Vector math**: Gram computed EXACTLY from low-rank factors (no ΔW materialization →
  no OOM, no truncation); verified == brute force to 4e-16. Dropped the broken random
  projection; MDS/arithmetic from the exact Gram.
- **Held-out honesty**: eval persists `ep_range`/`split`/`episodes_used`; every task is
  genuine held-out (n=30); fail-loud if FFT base missing.
- **Seed confound**: added bottle30s1 control; report cross-task cosine relative to the
  same-task ceiling. n=4 material/size "tests" dropped → descriptive only.

## Eval (base FFT vs FFT+LoRA, held-out, paired)
`eval_train_endhorizon.py --ep-range` on real LeRobot frames, open-loop / teacher-
forced action accuracy (per-joint MAE @ horizon end + within-tol). SAME seed for the
(base, LoRA) pair → identical sampled frames → paired Δ. Primary: bottle held-out
100:299. Secondary: bear 334:349, hat 419:424 (small); duck = train-fit (labeled).

## Vector analysis (`analyze_lora_vectors.py`, local, no GPU)
Per LoRA: ΔW_site = A·B at every adapter site (rank-axis contraction).
- **Exact full-ΔW cosine** task×task (per-site accumulation; norm-weighted).
- **Per-tower / per-kind cosine** (VLM vs action expert; attn vs FFN) — *where* task
  identity lives.
- **MDS + hierarchical clustering** of cosine distance → does structure match
  semantics (plush {bear,duck?} vs rigid {bottle,hat?}; size)?
- **Shared-skill direction**: alignment of each task to the mean unit ΔW → common
  "pick-and-place" component vs task-specific residual.
- **Random-projected vectors** (block JL) → PCA-2D + skill-arithmetic
  (bottle35≈bottle100? analogies).
- Honest n=4 caveat on attribute contrasts.

## Execution order (gated on FFT-56k completion)
1. FFT-56k finishes (step 55999 on disk + HF). ← currently training (~step 9k)
2. Register `pi05_fftlora` (FFT_INIT_PARAMS=…/55999/params).
3. LoRA smoke (5 steps): validate init-off-FFT + ep-filter + shared norm stats + no aug + finite loss.
4. Launch the LoRA jobs (parallel 1-GPU): bottle100, bottle35, bear35, duck35, hat35 (+ bear50, hat40).
5. `eval_fftlora_compare.sbatch` → base vs +LoRA held-out.
6. `extract_lora_adapters.py` per LoRA → vectors/*.npz; `analyze_lora_vectors.py`.
7. Phase C FFT-vs-GSE/LoRA/2task benchmark (independent; `eval_fft_bench.sbatch`).
8. Write the publishable report (data-driven).

## LoRA base = the BEST FFT checkpoint, not blindly the 56k final
User directive: tune the LoRAs off the best-generalizing FFT checkpoint, in case the
final over-memorises. The config/smoke/eval all reference a stable `lora_base`
SYMLINK; we set it to the selected step after the sweep, so nothing re-registers.

## Post-FFT execution sequence (run when .../farm_fft_multiobject_robust_406/55999/params exists)
```
# 0. confirm FFT done (all of 8k,16k,24k,32k,40k,48k,55999 on disk + HF)
# 1. SELECT the best-generalizing FFT checkpoint — TWO sweeps, then combine (/goal):
#    (a) training-data episodes (the literal ask): per-checkpoint action accuracy on
#        the multiobject set the FFT trained on → FIT curve.
sbatch eval_fft_ckptgen.sbatch         # → ~/farm-train/fft_ckptgen/traindata-fft-*.json
#    (b) held-out eval_bench (generalization cross-check; separate recordings + OOD task):
sbatch eval_fft_bench.sbatch fftonly   # → ~/farm-train/fft_analysis/eval-{clean,robust}-fft_*.json
#    → cp BOTH sets of JSONs into analysis/fullFlagshipFFT/fft_sweep/, then combine:
python model/select_fft_base.py --indir analysis/fullFlagshipFFT/fft_sweep
#    → prints fit-vs-generalization table + curve fig + recommended step. The FFT
#      trained on ALL multiobject, so (a) rewards memorization (favours final);
#      (b) reveals if a later step over-memorised. Recommend best HELD-OUT step.
# 2. point lora_base at the chosen step (e.g. 40000) ON THE CLUSTER:
ln -sfn <best> ~/farm-train/openpi/checkpoints/pi05_farm_multiobject_fft/farm_fft_multiobject_robust_406/lora_base
# 3. LoRA smoke (gate) — bear30 non-zero-start, expect SMOKE_FFTLORA_OK + small trainable count
sbatch smoke_fftlora.sbatch
# 4. on smoke OK → launch all 6 LoRAs (each → own HF repo), built on lora_base
for t in bottle30 bottle100 bear30 duck30 hat30 bottle30s1; do sbatch train_fftlora.sbatch $t; done
# 5. full Phase C FFT benchmark (FFT vs GSE/LoRA/2task)
sbatch eval_fft_bench.sbatch
# 6. after LoRAs finish → base-vs-LoRA held-out eval (base = lora_base) + adapter extraction
sbatch eval_fftlora_compare.sbatch
sbatch extract_lora_vectors.sbatch
# 7. pull artifacts to laptop, then locally:
#    kubectl cp .../lora_vectors → analysis/fftLoRA_report/vectors ; kubectl cp .../fftlora_analysis → ...
python model/analyze_lora_vectors.py --indir analysis/fftLoRA_report/vectors --outdir analysis/fftLoRA_report
# 8. write the publishable report from the metrics + figures.
```
Concurrent-GPU note: fairshare drops while the FFT runs (saw a small job sit at
reason=Priority); the 6 LoRAs schedule fine once the FFT releases its 4 GPUs.

## Status
- _build_: all configs/sbatch/eval/analysis written, staged, syntax-checked, math
  verified (Gram == brute force 4e-16). Episode-filter (Subset) patch APPLIED;
  pi05_fftlora REGISTERED. 5-reviewer pre-run audit → 3 blockers + 9 majors ALL FIXED.
- _FFT-56k DONE_ (job 406, 8h31m, all 7 ckpts on HF). Checkpoint generalization sweep
  (470 training-data + 471 held-out eval_bench) → **best step = 55999** (monotonic
  held-out improvement, no over-memorisation; see fullFlagshipFFT/CHECKPOINT_SELECTION.md).
  **lora_base → 55999** set. LoRA smoke (475) running → then 6 LoRAs.
