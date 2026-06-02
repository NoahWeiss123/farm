# Real-episode action-prediction eval — Multiobject GSE π0.5

Does the policy's predicted joint angles actually match the **real demonstrated
movement**? This tests the `pi05_farm_multiobject_gse` checkpoint (**step-5999**,
VLA-GSE full fine-tune) on real recorded episodes — **in-distribution** (its own
training set) and **out-of-distribution** (an earlier held-out dataset). No
fabricated data: every number comes from a real demonstrated frame fed through a
real GPU inference (CS153 cluster H100s).

## The metric (open-loop / teacher-forced)
For each sampled frame the model sees the **real** observation (7-dim joint
state + base & wrist camera frames + task prompt) and outputs a 10-step absolute
joint chunk. We compare `pred[k]` to the real recorded `state[t+1+k]` — the exact
target openpi trains on (30 fps, so `pred[9]` = `state[t+10]` = **333 ms ahead**).
This measures single-shot **prediction fidelity**, not closed-loop task success.
768 random mid-episode frames per condition + dense whole-episode rollouts for
the trajectory overlays.

| condition | dataset | what it measures |
|---|---|---|
| **in-dist** | `NoahWeiss/farm_uf850_multiobject` (424 eps, 4 tasks, the training set) | fit / memorization |
| **OOD** | `NoahWeiss/farm_uf850_bottle` (200 eps, held out) | generalization. Its *"bottle off box → desk"* task is an **instruction the model never trained on**. |

## Headline result
| metric | in-dist | OOD |
|---|---|---|
| horizon-end MAE (333 ms) | **1.31°** | **8.82°** |
| frames within 5° | **99.7%** | **28.1%** |
| predicted-vs-real displacement (Pearson r) | **0.940** | **0.079** |
| motion-direction agreement (cosine) | **0.84** | **−0.01** (chance) |
| error growth across the chunk | **flat** (1.26→1.31°) | **grows** (6.40→8.82°) |

The model reproduces its training demos almost perfectly — tracking position,
direction, and velocity — but on held-out data the **kinematic structure
collapses**: it emits workspace-plausible poses that no longer track the real
motion. The absolute MAE *understates* this (71.7% of OOD frames land within 10°
by workspace prior alone); the displacement/direction correlations expose it.
**See `FINDINGS.md` for the full analysis** and `farm_gse_realdata_eval.pdf` for
the figure report.

## Files
```
raw/eval-indist.json, eval-indist-raw.npz   # in-dist summary + full pred/gt chunks + rollouts
raw/eval-ood.json,    eval-ood-raw.npz       # OOD    summary + raw arrays
metrics.json                                 # all derived metrics, both conditions
FINDINGS.md                                  # the written analysis (multi-agent, adversarially verified)
farm_gse_realdata_eval.pdf                   # assembled figure report
figs/traj_<cond>_<task>.png                  # predicted next-angle vs real movement, per joint + gripper
figs/disp_<cond>.png                         # predicted vs real displacement scatter (per joint, R²)
figs/horizon_<cond>.png                      # error-vs-horizon curve + accuracy CDF
figs/perjoint_<cond>.png                     # per-joint error, direction agreement, velocity match
figs/pertask_<cond>.png                      # per-task breakdown
figs/compare_indist_vs_ood.png               # the head-to-head
```

## Reproduce
```bash
# 1. GPU inference (cluster) — dumps JSON + raw NPZ. Same model, two datasets.
sbatch model/cluster/eval_train_endhorizon.sbatch                                   # in-dist (TAG defaults)
sbatch --export=ALL,TAG=ood,REPO_ID=NoahWeiss/farm_uf850_bottle \
       model/cluster/eval_endhorizon.sbatch                                         # OOD
#   (plain `sbatch eval_endhorizon.sbatch` also works — inner.sh defaults to the OOD config)

# 2. pull eval-*.json + eval-*-raw.npz into analysis/realdata_gse/raw/ (rename to eval-indist*/eval-ood*)
# 3. local analysis (no GPU):
python analysis/realdata_gse/make_analysis.py     # metrics.json + figs/
python analysis/realdata_gse/make_report_pdf.py   # farm_gse_realdata_eval.pdf
```
Eval driver: `model/cluster/eval_train_endhorizon.py` (samples real frames, runs
one inference each + dense rollouts, dumps everything to NPZ for offline plotting).
