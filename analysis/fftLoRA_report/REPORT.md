# Task-LoRAs as a Composable Skill Library over a Full Fine-Tuned π0.5 Policy

**FARM / UF850 · CS153 · 2026-06**

> **Abstract.** We fine-tune the π0.5 vision-language-action model end-to-end on a
> 424-episode, 4-task tabletop manipulation set (the *base policy*), then train one
> low-rank adapter (LoRA) per task on top of the **frozen** base — each a small
> weight delta ΔW = A·B that we treat as a reusable "skill." We ask three questions.
> (1) *Which base checkpoint generalizes best?* Sweeping all checkpoints on held-out
> data shows generalization improves monotonically to the final step — heavy
> domain-randomization augmentation prevented over-memorization. (2) *Does adding a
> task-LoRA improve that task on held-out episodes the LoRA never saw?* **Yes — ~37%
> mean horizon-end MAE reduction on every task (headline: bottle 0.96°→0.56° on 60
> held-out episodes), generalizing within-task.** (3) *Do similar tasks yield similar LoRA
> vectors — i.e. is there a usable "skill space"?* **Not in raw weights:** LoRA ΔW
> directions are dominated by the random init seed — the *same* task trained with a
> different seed is orthogonal (cosine 0.04), while any same-seed pair sits at ~0.62.
> A real but **minority** task signal emerges only after controlling for the seed
> (same-task residual cosine +0.26 above cross-task), and it concentrates in the
> **vision/language tower**, not the action expert. The methodological takeaway — a
> seed control is essential, or raw LoRA cosines mislead — is itself the contribution.
> Code, models (all on HuggingFace), and the exact protocol are released.

---

## 1 Introduction

Full fine-tuning (FFT) gives a VLA policy maximal capacity to fit a new embodiment
and task set, but the result is monolithic: every new task means retraining or
risking interference. An attractive alternative is a **skills library** — a frozen
generalist plus a small, swappable adapter per skill. Because a LoRA is literally an
additive low-rank matrix ΔW = A·B applied to the base weights, a task-LoRA is a
concrete, composable object. This raises a representation-learning question by analogy
to image embeddings (where "wearing a hat" is a direction in latent space): **do
LoRAs for similar manipulation tasks occupy similar directions in weight space?** If
so, skill-space has exploitable structure (interpolation, composition, retrieval).

We study this on a real UF850 arm dataset with four pick-and-place tasks that vary the
object — **bottle, stuffed bear, rubber duck, hat** — onto a box. Our contributions:

1. A **best-generalizing base** selected by a held-out checkpoint sweep (§3), not by
   training loss.
2. A **controlled LoRA protocol** (§4): every task-LoRA is trained identically (same
   rank, steps, LR, seed, **shared normalization**, no augmentation) off the *same*
   frozen base, with **only the adapters trainable** — so the per-task delta is
   *purely* the LoRA. Equal-size task sets + same-task/different-seed and
   same-task/different-data controls make the cross-task comparison clean.
3. A **held-out adapter benefit** measurement (§5) and an **exact skill-space
   geometry** analysis (§6), the latter computed from the low-rank factors without
   ever materializing the 3.3-billion-dimensional ΔW.

## 2 The base policy: a robust π0.5 full fine-tune

**Data.** `NoahWeiss/farm_uf850_multiobject`: 424 teleop episodes / 129,067 frames at
30 fps, 4 tasks (bottle 0–298, bear 299–348, duck 349–383, hat 384–423), single 6-DoF
arm + gripper, base + wrist cameras.

**Training.** π0.5 (PaliGemma-2B + a 300M action expert, ~3.3B params) fine-tuned
end-to-end (no LoRA, no frozen modules) on 4× H100 (FSDP-2 → 2 data-parallel
replicas), batch 32, **56,000 steps (~13.9 epochs)**, cosine LR 2.5e-5 (2k warmup),
EMA 0.999. For robustness we apply the same **heavy domain-randomization** used by our
GSE flagship — wide brightness/contrast/saturation **+ hue** jitter, per-channel gamma,
grayscale, gaussian blur, crop/rotate — plus **prompt-paraphrase** augmentation across
all four task strings.

**Result.** The run completed in 8h31m; training loss fell monotonically 0.078 →
~0.0007. All seven checkpoints (8k…55999) are public at
`NoahWeiss/farm_uf850_multiobject_fft_robust`. This is the *base policy* the skills
LoRAs are built on.

## 3 Which checkpoint generalizes best?

Because the FFT trained on *all* 424 episodes, accuracy on any of them measures
memorization, not generalization, and would trivially favor the last step. We
therefore swept every checkpoint two ways with a fixed sampling seed:

- **Fit** — open-loop action accuracy on 48 multi-object (training-distribution)
  episodes.
- **Generalization** — clean + 9 domain-shift conditions on `eval_bench` (15
  separately-recorded episodes / 5 tasks, **including a `bottle→desk` task absent from
  the training set** → genuinely held out).

| step | train° (fit) | **held-out° (gen.)** | held-out robust° |
|---:|---:|---:|---:|
| 8000  | 2.54 | 2.34 | 3.03 |
| 24000 | 1.55 | 1.95 | 2.39 |
| 40000 | 1.19 | 1.74 | 2.16 |
| 48000 | 1.06 | 1.73 | 2.15 |
| **55999** | **0.98** | **1.68** | **2.12** |

**Both fit and held-out generalization improve monotonically to 55999** — the
over-memorization we guarded against did not occur; the heavy augmentation + 4-task
diversity regularized the full fine-tune enough to keep generalizing to the end (the
held-out curve does flatten after ~40k, so 40k is ~equal for 30% less compute). The
train↔held-out gap stays modest (0.98° vs 1.68°). **The selected base is step 55999.**
(Full method + figure: `CHECKPOINT_SELECTION.md`, `clean_7_checkpoint.png`.)

## 4 Method: task-LoRAs as composable skills

Every task-LoRA is trained under an **identical protocol** so that the only variable
is the task data — a prerequisite for treating the adapters as comparable vectors.

- **Base.** `gemma_2b_lora` + `gemma_300m_lora` adapters initialized on the **frozen**
  FFT-56k base via `CheckpointWeightLoader` (the dense FFT weights load into the base
  slots; `lora_a/lora_b` stay at init). **Only the adapters are trainable** — we freeze
  *all* non-LoRA parameters (including the SigLIP vision tower and the action heads,
  which openpi's default LoRA leaves unfrozen), so the per-task delta is purely the
  low-rank matrix. Rank 16 (VLM) / 32 (action expert), scale α/rank = 1.
- **No augmentation.** `FARM_AUG_LEVEL=off` (no frame shifting) and a single fixed
  prompt per task (no paraphrase) — each LoRA sees one task, one prompt, clean frames.
- **Shared normalization.** Every LoRA reuses the FFT base's *full-set* norm-stats
  (verified identical by sha256), so adapters differ only in learned weights, not in
  input scaling.
- **Identical hyperparameters.** batch 32, 12,000 steps, cosine LR 1e-4 (500 warmup),
  seed 42, checkpoints every 3k. Each LoRA streams to its own public HF repo.
- **Episode subsets** are selected at load time by a frame-index `Subset` of the full
  dataset (the installed lerobot's `episodes=` does not re-index, so a naive subset
  would crash non-zero-start tasks — we validated the fix on bear 299:329 = 8238 frames).

**The LoRA set** (all off the same base; the equal-size n=30 seed-42 set is the
controlled vector comparison; bottle100 and bottle30-seed1 are controls):

| LoRA | train eps | seed | role |
|---|---|---|---|
| bottle30 / bear30 / duck30 / hat30 | 30 each | 42 | equal-size vector set; held-out tails for eval |
| bottle100 | 0:100 | 42 | held-out bottle eval (eps 100–298, 199 held out) + data-scaling vs bottle30 |
| bottle30-seed1 | 0:30 | 1 | same-task/different-init control (cosine baseline) |

## 5 Does a task-LoRA improve held-out task performance?

*Protocol.* For each task, base FFT (the selected step-55999) vs FFT+task-LoRA on
episodes **outside** the LoRA's training slice — paired (identical seeded frames),
open-loop horizon-end joint MAE. Primary: bottle held-out 100–298 (the bottle100 LoRA
trained on 0–99; the base FFT saw all of them during its own training, the LoRA did not).

| task | held-out window | n | base FFT° | + task-LoRA° | Δ (improvement) |
|---|---|---:|---:|---:|---:|
| **bottle** (LoRA on 0:100) | 100–298 | 60 | 0.96 | **0.56** | **−0.40° (−42%)** |
| bear | 329–348 | 19 | 0.92 | 0.63 | −0.29° (−32%) |
| duck | 379–383 | 5 | 1.15 | 0.66 | −0.50° (−43%) |
| hat | 414–423 | 10 | 0.95 | 0.64 | −0.31° (−33%) |

**Adding a task-LoRA improves held-out action accuracy on every task** — ~37% average
MAE reduction, including the headline **bottle case (0.96° → 0.56° on 60 held-out
episodes the adapter never saw)**. Crucially the LoRA **generalizes within-task**: it
sharpens the policy on *unseen* episodes of its object, not just memorizing its training
slice. (All conditions are already at 100% within-5° accuracy for both base and +LoRA, so
the gain is in fine-grained, sub-degree precision rather than coarse task success — the
base FFT already "succeeds," the LoRA tightens it.) Figure: `clean_1_skills_help.png`.

**Reading §5 with §6.** The functional benefit here is real and consistent across tasks,
even though §6 showed the LoRA *weights* are init-dominated. That is the central
reconciliation: a task-LoRA is a genuine, generalizing **functional** skill, but its
raw weight vector is not a faithful skill *coordinate*. A skills library should index/
compose skills by behavior (or under a shared init), not by raw weight similarity.
Caveats: duck (n=5) and hat (n=10) held-out sets are small; offline action error is a
proxy for closed-loop success.

## 6 The structure of skill-space

*Protocol.* We extract each LoRA's ΔW = A·B and compute the **exact** task×task Gram /
cosine directly from the low-rank factors (`<AᵢBᵢ,AⱼBⱼ> = Σ_lead (AᵢᵀAⱼ)⊙(BⱼᵀBᵢ)`,
verified == brute-force ΔW to 4×10⁻¹⁶, no 3.3B-dim materialization). Six LoRAs at the
matched, converged step 9000: the n=30 seed-42 set {bottle,bear,duck,hat} plus two
controls — **bottle100** (same task, more data) and **bottle30-seed1** (same task,
different random seed). All ΔW have near-identical Frobenius norm (~27.5), so geometry
is about *direction*, not magnitude.

**Full-ΔW cosine (the headline):**

| | bear | bottle100 | bottle30 | **seed1** | duck | hat |
|---|---|---|---|---|---|---|
| bear | 1.00 | 0.61 | 0.61 | **0.00** | 0.64 | 0.62 |
| bottle100 | 0.61 | 1.00 | **0.72** | **0.02** | 0.63 | 0.61 |
| bottle30 | 0.61 | **0.72** | 1.00 | **0.04** | 0.63 | 0.61 |
| **seed1** | 0.00 | 0.02 | 0.04 | 1.00 | 0.01 | 0.00 |
| duck | 0.64 | 0.63 | 0.63 | 0.01 | 1.00 | 0.62 |
| hat | 0.62 | 0.61 | 0.61 | 0.00 | 0.62 | 1.00 |

**Finding 1 — the random seed dominates LoRA weight geometry, not the task.**
Same-seed LoRAs sit at cosine ≈ 0.61–0.64 for *every* task pair; each is ~0.84 aligned
to a single shared direction. But the **same task trained with a different seed
(bottle30 vs bottle30-seed1) is orthogonal (0.04)**, aligning only 0.25 to that shared
direction. So the shared direction is an artifact of the shared initialization, **not**
a semantic "pick-and-place skill": LoRA's low-rank factorization is non-unique, and
independent runs of the *same* task converge to orthogonal weight directions. A naive
raw-cosine reading would have wrongly reported "all tasks ~0.62 similar"; the seed
control is what exposes this. (This is the single most important methodological result.)

**Finding 2 — a real but minority task signal, only visible after controlling for seed.**
At fixed seed, same-task/different-data (bottle30↔bottle100) cosine is **0.72 vs ~0.62
cross-task** (Δ≈0.10). Projecting out the shared-seed direction, the same-task residual
cosine is **−0.02 vs −0.28 cross-task** (Δ≈0.26). So task identity *is* encoded — in the
~16% of the adapter orthogonal to the seed direction — but it is a minority component.

**Finding 3 — where task identity lives: the vision/language tower, not the action
expert.** Splitting by tower (action expert = the `_1`-suffixed Gemma expert): the **VLM
adapters are more task-discriminative** (cross-task cosine 0.58, same-task 0.69) than the
**action-expert adapters, which are more shared** (cross-task 0.74, same-task 0.85). The
interpretation is clean: *what* object to grasp adapts mostly in the vision/language
adapters (objects differ visually/semantically → lower cross-task similarity), while the
*pick-and-place motor program* is largely shared across objects in the action expert.

**Finding 4 — no semantic object-clustering, no skill-arithmetic (n=4).** Cross-task
similarities are ~uniform — no plush{bear,duck} vs rigid{bottle,hat} or size grouping —
and task-difference vectors are mutually orthogonal (analogy cosines −0.02…+0.00). With
four distinct objects this is descriptive, not a powered test, but there is no evidence
for the "hat-vector" style structure in raw LoRA space at this scale.

Figures: `clean_2_fingerprint.png` (the headline seed-vs-task control), `clean_5_similarity.png`
(full task×task cosine), `clean_3_skill_map.png` (MDS map — the different-seed copy off on its
own), `clean_4_where.png` (VLM-tower vs action-expert cross-task similarity).

## 7 Discussion & limitations

**For a skills library, raw LoRA weights are not a faithful "skill vector."** Because the
low-rank factorization is initialization-dependent, two LoRAs that *do the same task* can
be orthogonal in weight space (Finding 1) — so weight-space similarity/retrieval/averaging
across independently-trained adapters is unreliable. Two routes make a skills library
well-posed: (a) **share the initialization** (e.g., fix/freeze A, or seed all skills
identically — as we did for the n=30 set) so weights live in a common frame and the task
signal (Findings 2–3) is exposed; or (b) compare/compose skills in **function/behavior
space**, not weight space. The encouraging signals for (a): with a shared seed there *is*
a recoverable task component, and it is anatomically sensible (object identity in the VLM,
motor program in the action expert).

*Limitations.* Offline action error is a proxy for closed-loop success (no on-arm rollout).
n=4 objects gives no power for attribute-level grouping. The seed control has a single
instance (bottle); replicating different-seed runs per task would sharpen the seed-vs-task
decomposition. The held-out adapter-benefit (§5) is already at 100% within-5°, so the gain
is sub-degree precision, not coarse success; duck/hat held-out sets are small (n=5/10).

## 8 Conclusion

Building a skills library as task-LoRAs over a frozen full-FT π0.5 base is mechanically
clean, cheap, and **functionally effective**: the base itself is strong (the full fine-tune
generalized monotonically to 56k with no over-memorization, §3, and beats GSE/LoRA/2-task on
the held-out bench, §B), and **adding a task-LoRA reliably improves that task on held-out
episodes (~37% MAE reduction, §5)**. But the central representation-learning question — *do
similar tasks yield similar LoRA vectors?* — has a nuanced, control-dependent answer:
**raw LoRA weight directions are dominated by random initialization; the task contributes a
real but minority signal that is only legible after fixing the init, and it concentrates in
the vision/language tower rather than the action expert (§6).** The reconciliation: a
task-LoRA *is* a genuine, generalizing functional skill, yet its raw weight vector is not a
faithful skill *coordinate*. So a skills library is promising — but it must index and
compose skills by behavior (or under a shared initialization), not by raw weight similarity,
and the seed-control methodology here is what separates the real signal from the artifact.

## Appendix B — the base FFT vs other fine-tuning techniques

Context for "why this base." On the held-out eval_bench (15 eps / 5 tasks incl. an OOD
`bottle→desk`), offline action MAE, the selected FFT-56k base vs the alternatives:

| Model | data | clean° | domain-shift (mean)° | covers all 4 objects? |
|---|---|---:|---:|:--:|
| **FFT-multiobj (this base)** ★ | 424 ep / 4 task | **1.68** | **2.12** | ✓ |
| GSE-multiobj (robust) | 424 ep / 4 task | 1.94 | 2.50 | ✓ |
| 2-task FFT (bottle) | 200 ep / 2 task | 5.57 | 7.03 | ✗ (bear 11.9°, hat 10.8°) |
| LoRA (1-task bottle) | 100 ep / 1 task | 11.40 | 11.66 | ✗ (bear 21.5°) |

The full fine-tune is the best all-rounder — it beats even GSE on **both** clean fit and
domain-shift robustness (a reversal of the 2-task-era finding, now that 4-task diversity +
heavy aug let full-FT capacity pay off without over-memorizing), and only the two
multi-object models cover all four objects. Full detail + figure: `BENCHMARK.md`,
`clean_6_benchmark.png`.

## Appendix C — artifacts

`REPORT.md` (this), `BENCHMARK.md` (App. B), `CHECKPOINT_SELECTION.md`
(§3); figures `clean_1_skills_help.png` (§5), `clean_2_fingerprint.png` / `clean_5_similarity.png`
/ `clean_3_skill_map.png` / `clean_4_where.png` (§6), `clean_6_benchmark.png` (§B),
`clean_7_checkpoint.png` (§3); the plain-language `RESULTS_EXPLAINED.md` + `farm_fftlora_report.pdf`; data `lora_vector_metrics.json`,
`phaseC_compare.json`, `eval/cmp-*.json`. Base + 6 LoRAs + dataset all public on HuggingFace.

---

*Reproducibility.* Base model, all 6 LoRAs, and the dataset are public on HuggingFace.
Configs, training/eval scripts, and analysis code are in `model/cluster/` and `model/`;
the exact run protocol and review history are in `analysis/fftLoRA_report/PLAN.md`.
