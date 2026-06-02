# AUDIT VERDICT — GSE Multiobject Run (job 190)

**Run:** `farm_gse_multiobject_robust_190` · config `pi05_farm_multiobject_gse`
**Checkpoint shipped:** step 5999 · **Repo:** `farm_uf850_multiobject_gse_robust`
**Audited:** 2026-06-01

---

## 1. Bottom line

The training run executed correctly — there is no mechanical fault, and the GSE
wiring matches intent. The shipped model is a clean, well-fit **specialist** that
reproduces all four objects' *training* frames to ~1.3° but **does not generalize
to held-out episodes** (OOD displacement r=0.08, direction cosine -0.01 = chance).
That gap is the heart of the verdict: **the run trained well; the model does not
generalize** — and because you trained on 100% of the data with no validation
split and selected step-5999 on a monotonically-falling *train* loss, the pipeline
had no signal to catch that or to pick a better checkpoint.

---

## 2. Clean bill of health (the run itself is not broken)

Everything mechanical checks out. Do not chase phantom training bugs:

| Aspect | Status | Evidence |
|---|---|---|
| Completion | Clean to step 5999, **one init, no resume/restart** | train-gse-multiobj-190.out |
| Numerical stability | **Zero** NaN/inf, zero Traceback/CUDA-error/OOM/Killed | log grep (hits were benign NCCL lines) |
| Loss curve | Smooth **monotonic** 0.0742→0.0041@1k→0.0013@5k→0.0011@5900, no spikes/plateau-rise | gse_curve_clean.json |
| Grad norm | Smooth decay 0.545→0.025 | steps.txt |
| Frozen backbone | param_norm Δ +0.87 (1771.37→1772.24) ≈ flat; trainable-param tree in log shows only adapters + action expert | log + curve |
| Hardware | All **6 GPUs** active, data-parallel, batch 192 (32/GPU) | sbatch + log |
| Checkpoints | All 6 (1000…5999, 13 GB ea., params present) saved locally **and** pushed to HF in ~37-40 s; pusher drained cleanly | push-gse-multiobj-190.out |
| Recipe sanity | SVD-init (step-0 ≈ base π0.5), cosine 6e-5→6e-6, warmup 300, AdamW clip 1.0, ema off — all reasonable for scale | config |

Two non-issues to put to rest: (a) the `v2.1→v2.0` dataset-revision fallback
(2 warnings, not 4) resolved to a valid version and trained fine — it's a hygiene
nit, not a fault; (b) the NCCL-interleaved log lines are cosmetic stdout
multiplexing — the deduped metrics are intact and internally consistent.

**Caveat on the "frozen backbone" proof:** the *direct* evidence is the trainable
param tree printed in the log (gse_gen/spec adapters, FFN LoRA, action expert) — not
the flat global param_norm. A near-flat global norm cannot by itself distinguish a
frozen backbone from a tiny-LR full FT (the huge frozen VLM dominates the norm). The
conclusion is right; cite the param tree, not the norm.

---

## 3. Red flags, ranked by severity

The serious problems are **substantive (generalization / methodology)**, not
mechanical. Ranked:

| # | Issue | Severity | Bug / Expected / Tradeoff | Evidence | Recommendation |
|---|---|---|---|---|---|
| 1 | **No held-out validation split anywhere.** Trained on 100% of data; only train loss logged; orbax wrote `metrics:{}` at every save, `best_fn=None`. The fit/generalization gap was invisible during the run. | **High** | Methodology gap | gse_curve_clean.json (monotonic, no val); config/sbatch grep for val/eval_every/holdout = empty; orbax metrics:{} | Add a held-out in-domain split + periodic per-object val MAE/direction-cosine during training |
| 2 | **Checkpoint selected on train loss alone.** Step-5999 shipped because train loss is lowest there; the other 5 checkpoints were never evaluated (every eval pins `/5999`). | **High** | Methodology gap (not a code bug) | every eval = `…robust_190/5999`; grep for /1000…/5000 finds only pusher lines | Eval all 6 checkpoints on a held-out set; 5999 may be the *most* over-fit |
| 3 | **OOD generalization collapses to chance.** Held-out `farm_uf850_bottle`: end-MAE 8.82° (vs 1.31° in-dist), disp r=0.08, dir cos -0.013, all per-joint R² negative (to -20.8), within-5° 28% vs 99.7%; OOD horizon error *compounds* 6.40→8.82 while in-dist stays flat. | **High** (but see note) | **Expected-behavior** (small-data FT + domain/instruction shift), not a recipe bug | metrics.json `ood` | Not fixable by a hyperparameter; needs data coverage in the deploy distribution (see §4/§5) |
| 4 | **8.9 epochs is high-side for an adapter FT whose stated goal is robustness**; train loss still dropping at 5999 with no early-stop signal. | **Medium** | Tradeoff | 6000×192/129,067 = 8.93 epochs | Try 4-6 epochs (3-4k steps) with held-out-based selection; cheaper and plausibly ≥ as robust. **Do not** attribute the OOD collapse to step count — unproven |
| 5 | **No class-rebalancing under 70/12/9/8 imbalance** (bottle 299 / bear 50 / hat 40 / duck 35; first 299 eps all bottle). The recipe fact is true. | **Low/Note** | Tradeoff | data manifest; no sampler reweighting in patch_openpi_config_pi05_gse_multiobject.py | Add weighted/balanced sampling **if** held-out per-object metrics show a gap — current evidence does **not** show one (see §4) |
| 6 | **"Heavy aug + prompt aug = robustness" is overclaimed if read as OOD robustness.** Aug only buys photometric invariance on *in-distribution* frames. | **Medium** | Expected-behavior | robustness block (in-dist bench): clean 1.90° → noise 4.60 / occlude 3.28 / blur 2.91 (graceful); same ckpt on true held-out OOD = chance | State the scope honestly; verify aug's value via the ablation you already ran (overnight) |
| 7 | **Headline MAE understates the OOD collapse.** 8.82° / 71.7% within-10° "looks moderate" while structure is gone (r=0.08). | **Note** | Expected-behavior | metrics.json ood acc_within_10 0.717 vs disp r 0.079 | Report direction-cosine + displacement r alongside MAE; never quote MAE alone |
| 8 | **Eval metric is open-loop teacher-forced single-shot**, not closed-loop task success. Even the "in-dist specialist works" framing rests on a proxy. | **Note** | Methodology caveat | eval_train_endhorizon (TRAINING frames); FINDINGS.md:138 | Treat in-dist 1.31° as a fit upper-bound, not proof of on-arm success |
| 9 | **Throughput ~115-137 samples/s (~19-23/GPU)** on 6×H100 — modest; *plausibly* loader/video-decode bound. Wall ~2.7h training window (excl. ~24 min norm-stats pass + decode warmup → ~3h total job). | **Low** | Tradeoff | 1.67 s/step (incl. ckpt blocks); steady-state ~1.4 s/it | On a shared cluster, 2-3 GPUs or pre-decoded frames would use budget better. The "GPUs idle" claim is a hypothesis — no GPU-util trace in the log |

Two **rejected/downgraded** claims you can ignore:
- The "**class imbalance causes minority-object bench collapse**" claim is **rejected** — it rests on misattributed numbers (see §4).
- "**Step budget causes the OOD collapse**" is **downgraded** — the collapse is domain + novel-instruction shift, not over-training; the *real* medium concern is the absence of validation-based selection (#1/#2).

**Reproducibility caveat:** `max_to_keep=1` + `keep_period=1000` means intermediate
local checkpoints were pruned; the HF push is the **sole** complete archive of all
six. One-line note for future runs — the pusher was a single point of failure.

---

## 4. The class-imbalance thread (reconciled honestly)

The dataset is genuinely imbalanced — bottle 299 eps (70%), bear 50 (12%), hat 40
(9%), duck 35 (8%), with the first 299 episodes all bottle. That is a real latent
weakness and there is **no** sampler reweighting in the config. **But the current
evidence does not convict it for the GSE model**, and the audit chain that claimed
it did was built on a misattribution worth flagging loudly:

- **In-dist (training frames), per-object end-MAE:** bottle **1.24°** (n=528),
  bear **1.42°** (n=132), hat **1.52°** (n=84), duck **1.45°** (n=24). All within-5°
  99-100%, dir-cos 0.79-0.87. Minority objects fit **as well as** bottle.
- **Bench (GSE, ckpt 5999, lines 169-183):** bottle **2.28-4.02°**, bear
  **1.25-1.40°**, duck **0.88-1.11°**, hat **0.98-1.13°**. Minority objects are the
  model's **best** classes — the *opposite* of starvation.
- **The "bottle 0.7° vs bear/hat 6-19°" gap is a different model.** Those numbers
  are the `EVAL full` section (config `pi05_farm_uf850`, ckpt `farm_uf850_pi05_113`),
  a 2-task **bottle-only full-FT that never trained on bear/hat/duck** — so of course
  it collapses on them. Do not cite them against GSE.

**The honest ambiguity (don't overclaim either way):** the bench set has ~9/15
episodes overlapping training (bottle eps dated 20260528, minority 20260530), and
which specific episodes are train-vs-held-out is not pinned down. So "no
bottle-vs-minority gap on bench" is partly a *fit* score, and the apparent minority
*advantage* could reflect which eps happen to be memorized. The duck in-dist number
also rests on only n=24. **Conclusion:** imbalance is a real structural risk to
watch, but on every clean measurement we have, the minority objects are fine — the
"starved minority generalization" thesis is currently **unsupported and
contradicted**. Resolve the ambiguity with a proper held-out per-object split, not
by assuming the worst.

---

## 5. Concrete recommendations

In priority order — the first two are cheap and close the biggest gaps:

1. **Add a held-out in-domain validation split** (e.g. hold out N episodes *per
   object*, stratified) and log **per-object** val end-MAE, displacement r, and
   direction-cosine periodically during training. This single change makes
   over-fitting and any minority gap visible mid-run and gives you a real
   selection signal. Report direction-cosine/r, never MAE alone.

2. **Evaluate all 6 saved checkpoints** (1000…5999) on that held-out set before
   anointing a flagship. Step-5999 was picked on a monotone train curve and may be
   the most over-fit; an earlier checkpoint (cf. model/FINDINGS.md's 5k/10k/15k
   guidance) plausibly generalizes better and is essentially free to test
   (forward-only). This is the single missing experiment.

3. **Disentangle the OOD probe before drawing conclusions from it.** The held-out
   set confounds two axes: (a) cross-collection scene/domain shift (different
   recording session — the known deploy-room mismatch), and (b) a **novel
   instruction** ("bottle off the box → desk", dir-cos -0.186) that was *never in
   the 4-task training set*. The trained-instruction OOD task (dir-cos +0.132) is
   also poor, so domain shift dominates — but report the two splits separately so
   the headline "does not generalize" isn't conflating an unseen instruction with
   true generalization failure. No checkpoint choice fixes (a); only adding deploy-
   distribution data does.

4. **Hold the rebalancing knob in reserve.** Do *not* add weighted sampling
   reflexively — the evidence doesn't currently justify it. Add it only if the
   held-out per-object metrics from (1) reveal a real minority gap. If they do,
   weighted sampling / minority oversampling / per-class batch balancing are the
   levers.

5. **Verify aug actually helps via the ablation you already have.** You ran the
   overnight aug-vs-no-aug variants — use them to state aug's contribution honestly
   rather than asserting "robustness delivered." On current evidence, heavy aug buys
   graceful *photometric* degradation on seen scenes only; it does nothing for the
   held-out collapse.

6. **Reconsider epochs / add early-stop.** 8.9 epochs with no early-stop is more than
   this 424-ep set likely needs for a robustness-oriented adapter FT. With a val
   signal in place, 4-6 epochs + held-out-based selection is the cheaper, likely-no-
   worse default.

7. **(Minor) Efficiency + reproducibility:** for the next run, prefer 2-3 GPUs or
   pre-decoded frames on the shared cluster (loader-bound, not compute-bound), pin
   the dataset revision explicitly, and don't rely on `max_to_keep=1` + a single
   pusher as the only archive of intermediate checkpoints.

---

**One-line summary:** The run is mechanically clean and the recipe is sound; the
deliverable is an in-distribution specialist that does not generalize, shipped via
train-loss-only checkpoint selection with no validation set — fix the
*methodology* (held-out split + checkpoint sweep), not the *machinery*.
