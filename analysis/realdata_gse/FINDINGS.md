# Does the multiobject GSE π0.5 policy predict real demonstrated motion? In-distribution vs. out-of-distribution

**Model under test:** `pi05_farm_multiobject_gse`, checkpoint `step-5999` (VLA-GSE full fine-tune of π0.5 on `NoahWeiss/farm_uf850_multiobject`, 424 eps, 4 tasks: bottle/bear/duck/hat → box).

## TL;DR

We measured open-loop, teacher-forced single-shot prediction fidelity: the model always sees the **real** recorded observation (7-DoF joint state + base/wrist cameras + task prompt) and emits a 10-step absolute-joint chunk; we score `pred[k]` against the real recorded `state[t+1+k]` (33 ms/step, horizon end = 333 ms ahead). On its **own training set** (Condition A, in-distribution) the policy nearly retraces the demonstrations: 1.31° horizon-end MAE, a flat error-vs-horizon curve, displacement Pearson **r = 0.940**, and direction cosine **0.84**. On **held-out** episodes (Condition B, OOD: `NoahWeiss/farm_uf850_bottle`, 200 eps never in training) the absolute MAE rises "only" 6.7× to 8.82° and 71.7% of frames stay within 10°, which looks like partial success — but every metric that measures whether the prediction tracks the *demonstrated motion* collapses: displacement r → **0.079** and direction cosine → **−0.013** (chance), with all per-joint R² turning negative. The reconciliation is the finding: within a bounded joint workspace, emitting any plausible in-distribution pose keeps absolute error small, while the motion-structure metrics (which subtract out that static prior) expose that no scene-conditioned visuomotor signal transfers. This is a generalization **failure** wearing the costume of partial success.

---

## 1. What was measured

**Metric (open-loop / teacher-forced).** For every sampled frame, the model receives the **real recorded observation** at time `t` (7-dim joint state, base + wrist camera frames, task prompt) and outputs a 10-step chunk of absolute joint targets. We compare each predicted step `pred[k]` to the **real recorded** state `state[t+1+k]` — this is exactly openpi's training target. The data is 30 fps, so step `k` is `33·(k+1)` ms ahead; the horizon end `pred[9] = state[t+10]` is 333 ms ahead.

This measures **prediction fidelity**, *not* closed-loop task success. The model never sees its own previous outputs; every prediction is re-grounded on a real observation, so errors do not compound across the metric itself (only within a single 10-step chunk). Sampling: seed 0, 768 random mid-episode frames per condition over 64 episodes, window `[0.25, 0.75]` of each episode (the high-motion middle, where a static pose prior helps *least*), plus dense whole-episode rollouts for the trajectory figures.

**Step clock.** `pred[k] ↔ state[t+1+k]`, `k = 0…9`. First-step = 33 ms, horizon-end = 333 ms.

**The two datasets.**

- **Condition A (in-distribution, FIT):** `NoahWeiss/farm_uf850_multiobject` — the model's **own** training set (424 eps; 4 tasks bottle/bear/duck/hat → box).
- **Condition B (out-of-distribution, GENERALIZATION):** `NoahWeiss/farm_uf850_bottle` — 200 held-out episodes from a *separate* collection session, never in multiobject training. Two task strings:
  1. *"Picking up the bottle and placing it on the box"* — the **same instruction string** as a multiobject training task, but different/held-out episodes (a same-task, cross-collection shift).
  2. *"Picking up the bottle off of the box and putting it on the desk"* — a **reverse task whose instruction the multiobject model never trained on** (unambiguous novel-instruction OOD).

**Why this is a valid OOD probe.** Same UF850 arm, same 7-DoF absolute joint space, same 30 fps, same base+wrist cameras as the training data, so the observation/action interface matches deployment. The never-trained reverse instruction guarantees at least one split is unambiguously novel; the held-out episodes guarantee that even the seen-instruction split tests cross-collection novelty rather than memorized frames. Normalization is handled correctly (see §7).

---

## 2. In-distribution results (fit)

On its own training set the policy reproduces the demonstrated motion almost exactly. This is **fit / memorization**, not generalization — and it is the floor against which Condition B is read.

| Metric (Condition A) | Value |
|---|---|
| First-step MAE (33 ms) | **1.26°** |
| Horizon-end MAE (333 ms) | **1.31°** |
| Chunk-mean MAE | 1.25° |
| Within 2° / 5° / 10° | **90.8% / 99.7% / 100%** |
| Displacement Pearson r (333 ms) | **0.940** |
| Per-joint displacement R² | [0.83, 0.93, 0.76, 0.89, 0.87, 0.85] (all +) |
| Motion-direction sign agreement (per joint) | [0.83, 0.97, 0.88, 0.86, 0.86, 0.88] |
| Mean 6-vector displacement cosine | **0.84** (median 0.93) |
| Velocity-profile Pearson r | 0.842 |
| Gripper within 0.1 | 99.0% |

**Error does not compound across the chunk.** The error-vs-horizon curve is flat: `[1.26, 1.24, 1.22, 1.21, 1.22, 1.22, 1.24, 1.26, 1.27, 1.31]°` — the 333 ms-ahead step is only 0.05° worse than the first step. This means the 10-step chunk has **uniform per-step prediction accuracy** against the teacher-forced targets (no compounding *in prediction*). It does **not** imply anything about closed-loop execution reliability or task success, which this metric cannot measure. See `figs/horizon_indist.png`.

**The model reproduces motion, not a static held pose.** Displacement r = 0.940 with all-positive per-joint R², per-joint direction agreement well above the 50% chance baseline, mean displacement cosine 0.84, and velocity r = 0.842 together show the policy correctly commands where each joint is heading and roughly how fast. The displacement scatter (`figs/disp_indist.png`) is a tight diagonal cloud.

**Per-task structure** (`figs/pertask_indist.png`). Bottle dominates the sample and fits best; rarer tasks fit slightly worse, roughly in inverse-frequency order, but all stay under 1.6° with 99–100% within 5%:

| Task | Horizon-end MAE | Within 5% | n |
|---|---|---|---|
| bottle → box | 1.24° | 99.8% | 528 |
| bear → box | 1.42° | 99.2% | 132 |
| duck → box | 1.45° | 100% | 24 |
| hat → box | 1.52° | 100% | 84 |

Gripper open/close is reproduced almost perfectly (99.0% within 0.1), consistent with the clean square-wave tracking in the trajectory figures (`figs/traj_indist_*.png`), which show the 10-step-lookahead and next-angle predictions overlaying the real demo across all 6 joints + gripper with only hairline deviation.

---

## 3. Out-of-distribution results (generalization)

On held-out episodes the same metric collapses on every motion-structure axis, even as absolute MAE stays deceptively bounded.

| Metric | In-distribution (A) | Out-of-distribution (B) |
|---|---|---|
| First-step MAE | 1.26° | **6.40°** |
| Horizon-end MAE (333 ms) | 1.31° | **8.82°** (6.7×) |
| Chunk-mean MAE | 1.25° | 7.59° |
| Within 2° / 5° / 10° | 90.8% / 99.7% / 100% | **2.2% / 28.1% / 71.7%** |
| Error-vs-horizon shape | FLAT (1.26→1.31) | **GROWS** (6.40→8.82) |
| Displacement Pearson r | **0.940** | **0.079** |
| Per-joint displacement R² | all positive | **[−8.6, −2.7, −9.6, −20.8, −4.2, −11.9] (all negative)** |
| Mean displacement cosine | **0.84** (med 0.93) | **−0.013** (med −0.037) |
| Direction sign agreement (per joint) | [0.83…0.88] | **[0.47, 0.49, 0.45, 0.52, 0.52, 0.54] (~chance)** |
| Velocity-profile r | 0.842 | 0.162 |
| Gripper within 0.1 | 99.0% | 84.0% |

OOD error-vs-horizon: `[6.40, 6.64, 6.91, 7.17, 7.43, 7.71, 7.99, 8.28, 8.56, 8.82]°` (`figs/horizon_ood.png`).

### The central insight: absolute MAE understates the failure

The "only 6.7× / 71.7% within 10°" headline is misleading. The reconciliation:

- **Absolute MAE is flattered by a bounded workspace prior.** Because the actions are *absolute joint targets*, a model that merely emits a plausible in-workspace pose lands within ~10° of any realistic UF850 reach pose by geometry alone. So bounded absolute error reflects the workspace, not tracking skill. *(This workspace-prior mechanism is the best-supported interpretation of the data — it is consistent with the displacement/direction collapse and the disp_ood scatter clustering near the origin; we did not measure an explicit static-pose baseline MAE, so it is an interpretive hypothesis, not a directly measured artifact.)*
- **The prior-free metrics expose the collapse.** Displacement Pearson r falls **0.940 → 0.079** and the 6-vector direction cosine falls **0.84 → −0.013** — exactly chance. The model does not even reliably know which **way** each joint should move (per-joint sign agreement ~0.50).
- **The decisive evidence is the all-negative per-joint R²** `[−8.6, −2.7, −9.6, −20.8, −4.2, −11.9]`. A negative R² means the model's commanded displacement is a *worse* predictor of the real demonstrated displacement than a constant "no-motion" baseline. The predictions are **uncorrelated with the true motion** (r = 0.079, chance) and their magnitude/offset is wrong enough to score below the mean baseline — this is the signature of emitting a near-static prior trajectory against varying targets, **not** anti-correlation. (The negative magnitudes themselves, e.g. −20.8 on j4, are variance-sensitive and should be read as "far below the mean baseline," not as calibrated effect sizes.)
- **The horizon curve flips meaning.** In-dist flat = the motion was actually captured; OOD growing = the model has no correct velocity/direction to extend, so it drifts further from truth the longer it extrapolates. Velocity match drops 0.842 → 0.162.

**Verdict:** low MAE + chance direction = the model outputs a plausible-looking but scene-blind pose. The 71.7%-within-10° number must always be read alongside the motion-structure collapse; it is not 71.7% useful predictions.

---

## 4. Instruction-novelty split

Splitting Condition B by task separates cross-collection episode/visual novelty from instruction novelty (`figs/pertask_ood.png`):

| OOD task | Horizon-end MAE | Within 5% | Direction cosine | n |
|---|---|---|---|---|
| "bottle → box" — **seen instruction**, held-out episodes | 9.17° | 35.2% | **+0.13** | 420 |
| "bottle off box → desk" — **never-trained instruction** | 8.40° | 19.5% | **−0.19** | 348 |

Two observations:

1. **The seen instruction buys only a faint directional prior.** The familiar prompt string keeps direction cosine weakly positive (+0.13) — a residual of correct motion sense carried by the language token — while the never-trained reverse instruction falls to **near-chance (slightly negative, −0.19)**. The reverse split is genuinely negative on direction, but −0.19 is near-chance, not strong reliable anti-correlation; we do not claim the model reliably points the wrong way. Both are far below the in-dist 0.84.

2. **Accuracy–MAE inversion.** The seen-instruction task has *higher* absolute MAE (9.17 vs 8.40°) yet *better* within-5% (35.2% vs 19.5%) and a positive direction cosine. MAE and direction quality are decoupled: absolute error is governed by workspace geometry while the seen instruction still buys partial directional correctness. (Note: the per-task column is the source's "within-5%" metric, not the 5-degree threshold.)

Critically, the seen-instruction split being **worse on absolute MAE** (9.17 vs 8.40°) rules out "it failed only because the instruction was novel." The failure is a real scene/collection distribution shift, not merely unseen language — see §7.

---

## 5. Per-joint / kinematic structure

The same error structure appears in both conditions, scaled up under OOD (`figs/perjoint_indist.png`, `figs/perjoint_ood.png`).

| Joint | In-dist horizon-end MAE | OOD horizon-end MAE |
|---|---|---|
| j1 | 0.92° | 6.32° |
| j2 | 0.86° | 6.83° |
| j3 | 1.57° | 10.29° |
| j4 | **1.85°** | **14.87°** |
| j5 | 1.26° | 6.84° |
| j6 | 1.38° | 7.77° |

- **j4 is the hardest joint and j3 second in both conditions**, with the base pair (j1, j2) cheapest as a group. The *worst* ranking (j4 > j3, then j6, j5) is preserved across conditions; the base pair stays cheapest but its internal order flips (in-dist cheapest is j2 then j1; OOD cheapest is j1 then j2), so the ranking is preserved as a structure, not byte-for-byte identical.
- The OOD blowup is largest exactly on the joints that carry the most predictive burden (j4 1.85 → 14.87°, j3 1.57 → 10.29°), while the base joints that move through a narrow, repeatable arc stay relatively cheap (j1 6.32, j5 6.84°). This is consistent with the workspace-prior reading: joints that barely move keep small absolute error; joints that must execute the actual reach/place motion are where tracking failure becomes visible. *(Caveat: per-joint MAE is in raw degrees and is not normalized by each joint's range of travel, so "worst in MAE" is not identical to "worst in normalized fidelity"; the all-negative OOD R² nonetheless confirms a genuine collapse on j3/j4.)*
- The preserved rank, plus j4 being worst in both, argues the OOD failure is a **uniform scaling of the same error structure** — not a new failure mode, a single pathological joint, or a unit/indexing bug.

---

## 6. Caveats & confounds

- **The "bottle → box" OOD split shares its instruction string AND object with a trained multiobject task** (the training set contains 299/424 bottle→box episodes). Only the *recorded episodes* are held-out (a separate `farm_uf850_bottle` collection). It is therefore a **cross-collection / same-task shift, not pure novel-task OOD.** The clean, unambiguous novel-instruction evidence is the reverse "bottle → desk" split (n = 348, 8.40°, dir-cos −0.19), which fails comparably. Because the seen-instruction split is in fact *worse* on absolute MAE (9.17 vs 8.40°), the conclusion holds regardless: the failure is not explained by unseen language. (On *direction*, the seen split is slightly less collapsed, +0.13 vs −0.19, but both are far below the in-dist 0.84.)
- **Teacher-forced ≠ closed-loop.** Every prediction is conditioned on the real recorded observation, so this measures single-shot prediction fidelity, not on-arm task success, grasp completion, or recovery. No success-rate claim can be made from these numbers. **Closed-loop OOD would almost certainly be *worse*:** with direction cosine at chance, the policy doesn't know which way to move from a true state, so once its own wrong actions push the arm off-manifold the errors would compound without ground-truth re-grounding. The open-loop 8.82° is therefore an **optimistic floor** on OOD badness — this is an inference, not a measured number.
- **Workspace-prior effect on absolute MAE.** Absolute-joint MAE is flattered by the workspace prior on *both* conditions, so the 6.7× ratio *understates* the collapse. The prior-free metrics (displacement Pearson 0.940 → 0.079, direction cosine 0.84 → chance) are the load-bearing generalization evidence. We did not include an explicit static-pose baseline MAE to quantify exactly how much of 8.82° is prior vs. residual signal, though the all-negative R² implies the model is below that baseline on displacement.
- **Normalization is valid, not a handicap.** Both conditions use the identical checkpoint (`…/5999`) and config (`pi05_farm_multiobject_gse`), so `create_trained_policy` loads the **same** norm-stats (computed from multiobject training) for both — exactly what happens at deploy. Both datasets are the same UF850, same 7-DoF absolute joint space, same 30 fps, same base+wrist cameras. Normalizing OOD data with training stats is correct, not an unfair inflation. Prompts are read verbatim (`prompt_from_task=True`) with augmentation disabled at inference, so there is no prompt mismatch — the "bottle → box" OOD instruction is byte-identical to a trained one.
- **In-distribution is the model's OWN training set** (FIT, with likely memorization given r = 0.94 and the flat horizon curve), so it is an upper bound on fit, not a held-out same-distribution validation score. A held-out in-domain split would be a stronger anchor.
- **Sample asymmetry / imbalance.** Matched seed (0), frame count (768), episodes (64), window `[0.25, 0.75]`, and horizon (10) across conditions. But in-dist mixes 4 tasks (bottle-dominated, 528/768; duck n=24 and hat n=84 are thin/noisier) while OOD is 2 bottle tasks. Since bottle is the in-dist *strongest* task (1.24°), this if anything makes the in-dist baseline tougher to beat, not easier.
- **Per-project context:** prior diagnosis attributed live failures partly to a deploy-room/camera domain shift. The seen-instruction split being worse than the novel reverse task is consistent with a scene/collection domain gap between collections — i.e., part of "OOD" here is environment shift, not purely task/semantic generalization.

---

## 7. Bottom line

On its 4-task multiobject training set, the GSE full fine-tune of π0.5 fits the demonstrated trajectories tightly — ~1.3° non-compounding prediction error out to 333 ms, displacement r = 0.940, direction cosine 0.84, gripper 99.0% — but on held-out episodes its motion prediction collapses to chance: displacement r = 0.079, direction cosine −0.013, all-negative per-joint R², and a now-growing error-vs-horizon curve. The bounded absolute MAE (8.82°, 71.7% within 10°) is best explained by the model staying inside the learned, low-dimensional joint workspace while retaining **no transferable, scene-conditioned motion policy**. The familiar instruction string buys only a faint directional prior (+0.13 seen vs −0.19 never-trained), confirming the model leans on memorized training-episode associations rather than generalizing from the visual scene.

**This corroborates the project's prior findings:** a small-multi-task full fine-tune fits its training manifold extremely tightly but learns a **workspace prior, not a generalizable visuomotor skill.** When held-out observations fall just off that manifold, the policy snaps to a plausible-but-scene-blind pose — which is exactly why live OOD deployment failed, and why the chance-level direction cosine (the prior-free signal), not the deceptively bounded MAE, is the number that would govern closed-loop behavior. The aggregate side-by-side is summarized in `figs/compare_indist_vs_ood.png`.
