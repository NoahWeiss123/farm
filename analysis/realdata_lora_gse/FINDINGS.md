# base GSE vs GSE + LoRA — does adding a bottle LoRA help?

We compare the **GSE multiobject model** ("**base GSE**", `pi05_farm_multiobject_gse`, step-5999)
against the same model with a **bottle LoRA** trained on top of it ("**GSE + LoRA**",
`pi05_farm_bottle_lora_gse`, a LoRA on `multiobject[0:100]` initialized off base GSE via
`GSEMergeWeightLoader`). Metric = open-loop, teacher-forced single-shot prediction (the model sees
the real observation, outputs a 10-step absolute-joint chunk; we score `pred[k]` vs the real recorded
`state[t+1+k]`; horizon end = 333 ms ahead). This is **prediction fidelity, not closed-loop task
success.**

## TL;DR
- **On the 100 bottle episodes the LoRA specialized on, GSE + LoRA is the clear winner.** Across 15 of
  those episodes it reproduces the demos near-perfectly and *uniformly* — **0.53°** average (range
  0.43–0.62°, **no outliers**) vs base GSE's **1.09°**. The LoRA tightens bottle fit ~2×.
- **But the LoRA bought that by over-specializing.** On *other* bottle episodes (`multiobject[100:299]`,
  same camera) GSE + LoRA degrades to **8.62°**, while base GSE still handles them at **1.31°** — those
  episodes were in base GSE's training, so this is a fit ceiling, but the point stands: base GSE knows
  the whole bottle distribution; the LoRA traded that breadth for a tighter fit on its 100.
- **Net:** add the LoRA if you only care about those 100 episodes' exact motions; keep base GSE if you
  need bottle competence beyond them.

## 1. Within-training fit — 15 episodes  (`figs/per_episode_fit.png`, `figs/traj_fit15_gse.png`)
15 episodes drawn from the 100 the LoRA trained on (30 frames each), scored for both models.

| model | avg per-episode MAE | best ep | worst ep | outliers (>μ+2σ=0.65°) |
|---|---|---|---|---|
| base GSE | 1.09° | 0.90° | 1.33° | — |
| **GSE + LoRA** | **0.53°** | 0.43° | 0.62° | **none** |

- **GSE + LoRA reproduces every one of these episodes near-perfectly and consistently** (0.43–0.62°,
  displacement r 0.984, direction cosine 0.954). The fit is uniform — **no episode is a poor-fit
  outlier**; it memorized all of them evenly.
- **base GSE is looser (~1.1°)** on the very same episodes — it splits capacity across four objects, so
  the bottle-specialized LoRA fits tighter. Both have zero outliers; GSE + LoRA is simply ~2× closer.
- The trajectory overlay (`traj_fit15_gse.png`) shows the LoRA's commanded next-angle sitting on top of
  the real demo across all 6 joints + gripper.

## 2. The cost: generalization beyond the 100  (`figs/headtohead.png`, `figs/fit_vs_heldout.png`, `figs/ckpt_sweep.png`)
Probe = unseen bottle episodes `multiobject[100:299]` — same camera + same →box task. (The separate
`farm_uf850_bottle` set is a different camera config → excluded as a domain shift, not a generalization
signal.)

| held-out [100:299] | MAE ↓ | within-5° ↑ | displacement r ↑ | direction cos ↑ |
|---|---|---|---|---|
| base GSE* (fit ceiling) | **1.31°** | 100% | 0.934 | 0.833 |
| **GSE + LoRA** | 8.62° | 34% | 0.236 | 0.357 |

- **The LoRA over-specialized.** GSE + LoRA collapses from its near-perfect fit (0.53°, disp r 0.98) to
  8.62° / disp r 0.24 on these episodes — it memorized its 100 rather than learning a broad bottle
  policy. Direction cosine 0.36 is above chance (0) but far below the 0.95 fit: partial generalization.
- **base GSE handles the same episodes at 1.31°** because it trained on all 299 bottle episodes. *This is
  a fit ceiling, not a fair held-out comparison* — but it shows base GSE retains competence across the
  whole bottle distribution that the LoRA gave up.
- **Checkpoint selection** (`ckpt_sweep.png`): if you do use the LoRA, **step-2000 generalizes best**
  (8.36° / 38%), not the default step-9999 — "later ≠ better" (held-out MAE peaks at step-6000), the
  over-fit signature.

## 3. Caveats
- **Open-loop, teacher-forced single-shot prediction — NOT closed-loop task success.**
- **Absolute MAE is flattered by the bounded joint workspace;** displacement r / direction cosine are
  the load-bearing signals.
- **base GSE's held-out rows are FIT ceilings** (it trained on these episodes), not a clean held-out
  comparison — included to show retained breadth, not to claim base GSE "generalizes" better per se.
- **No seeds / no error bars / single probe.**

## Bottom line
**Adding a bottle LoRA to the GSE model makes it a near-perfect, uniform reproducer of the 100 episodes
it specialized on (~0.5° vs base GSE's ~1.1°), but it loses competence on bottle episodes outside that
set (8.62° vs base GSE's 1.31° fit).** Use GSE + LoRA when you need the tightest fit on those specific
demonstrated motions; keep base GSE when you need to cover the broader bottle distribution. If you ship
the LoRA, ship **step-2000**.
