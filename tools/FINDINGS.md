# π0.5 FARM policy — performance diagnosis & action plan

Audit done 2026-05-29 against the deployed full-FT `pi05_farm_uf850`
(step-19999). Symptoms reported: ~30 % success on the **trained** bottle
tasks, **complete failure** outside training, and **jittery** motion.

Everything below was found by static review + local data analysis
(`tools/analyze_dataset.py`) — no GPU required. The training-recipe changes
are validated for *syntax* only; they must be confirmed on the cluster.

## TL;DR

| Lever | Effort | Expected impact |
|---|---|---|
| **1. Pick an earlier checkpoint** (5k/10k vs 19999) | none (ckpts exist) | ↑ generalization, ↓ canned-motion, ↓ jitter |
| **2. LoRA fine-tune instead of full FT** | 1 GPU run | ↑↑ OOD generalization, ↓ memorization |
| **3. Prompt/paraphrase augmentation** | data/transform | ↑ language generalization |
| **4. More diverse data** (more objects/verbs) | data collection | the only real fix for *broad* generality |
| **5. Tuned RTC alone for smoothness** | serve flag | ↓ jitter at chunk seams |

## What is NOT wrong (ruled out)

Run `python tools/analyze_dataset.py --dataset datasets_lerobot/farm_uf850_bottle --raw Dataset3`:

- **Action labels are correct.** `action[t] == state[t+1]` to 0.00e+00 — no
  off-by-one, absolute next-state joint targets exactly as the config expects.
- **The data is smooth.** Joint-velocity sign-flip rate is 3–5 %/step and the
  static-frame fraction is ~5 %. A jerky demo would flip much more often.
  **⇒ the jitter at inference is NOT learned — it is a model/serving artifact.**
- **Cameras are distinct and sane** (base ≠ wrist), solid 30 Hz capture
  (occasional ~100 ms hiccup, absorbed by the fixed-fps export grid).
- **The gripper actuates properly** — ~2 grasp/release transitions per episode;
  the grasp band sits near ~0.3 (not 1.0), which the eval/back-end handle
  correctly (continuous 0–1 → SDK 0–850).
- **Train/serve transform parity holds** — serving reuses the exact
  `LeRobotFarmDataConfig` (LiberoInputs/Outputs, `use_delta_joint_actions=False`,
  openpi's own norm stats from `compute_norm_stats.py`). No silent mismatch.

(Minor: the LeRobot `meta/stats.json` image means look buggy — identical across
all 6 RGB channels. Harmless: openpi recomputes its own norm stats and
normalizes images inside the vision tower, so it never reads those.)

Minor data hygiene (cheap, low impact): one episode is only 31 frames (likely
an aborted demo — consider dropping it), and episodes carry ~8 leading idle
frames on average (the arm sitting still before motion). Trimming leading idle
in `export_lerobot.py` would slightly sharpen start-of-episode behavior. Image
exposure is excellent and consistent (brightness σ≈0.017, no dark/bright
outliers), so lighting/camera is not a factor.

## Root causes (ranked)

### 1 & 2 — Full fine-tune of 3.3B π0.5 on 2 tasks → memorization + forgetting
The live config trains **all** params (PaliGemma 2B + Gemma 300M action
expert) for **~21 epochs** on **2 near-inverse bottle tasks**. On a dataset
this small and narrow, full FT (a) memorizes the two demonstrated trajectories
— hence "does specific motions that don't align to the current object pose" —
and (b) overwrites the web-scale + cross-embodiment priors that give π0.5 its
generalization, so anything outside the 2 trained prompts collapses.

**Fix A (free, do first): checkpoint selection.** Earlier checkpoints have
memorized less. They already exist on HF. Serve each and compare with
`offline_eval.py` on **held-out** frames (and ideally novel bottle positions):

```bash
# on the login pod, for STEP in 4999 9999 14999 19999:
hf download NoahWeiss/farm_uf850_pi05 --include "step-$STEP/*" \
    --local-dir ~/farm-train/checkpoints/pi05_step$STEP
# serve it (edit serve_pi05.sbatch --policy.dir to the step dir), then:
uv run python ~/farm-train/offline_eval.py ~/farm-train/eval_episode
# pick the lowest joint-MAE checkpoint that ALSO behaves on novel positions,
# not necessarily 19999.
```

**Fix B (1 GPU): LoRA fine-tune.** Added as `pi05_farm_uf850_lora`
(`tools/cluster/patch_openpi_config_pi05_lora.py`, registered by `setup.sh`).
Freezes the backbone, trains low-rank adapters → preserves the base. Fits on a
single H100, so it respects the shared-cluster gpu:1 norm. Same serve/eval path
(action_horizon=10, absolute actions) — no client changes.

```bash
# after setup.sh has registered it:
sbatch tools/cluster/train_pi05_lora.sbatch        # → NoahWeiss/farm_uf850_pi05_lora
```

The full-FT config is unchanged and still the default — LoRA is opt-in so you
can A/B them. If LoRA *underfits* the trained tasks (offline_eval MAE stays
high), the backbone genuinely needs to move for this embodiment → go back to
full FT but with an earlier checkpoint and/or fewer steps.

### 3 — Prompt overfitting (2 fixed strings), and the tasks alias at the boundaries
Training sees only the 2 exact task strings, so the language pathway
degenerates and novel phrasings are out-of-distribution.

This is *especially* damaging here because **the two tasks are visually
ambiguous at their endpoints** (from `/tmp` deep analysis, now folded into the
diagnosis): every episode starts AND ends with the gripper open (~0.001), start
poses overlap (|Δmean|/σ < 0.4 on all joints), and the end-state of task 0
(bottle on the box, gripper open) is essentially the *start-state* of task 1.
So "place it on the box" vs "move it to the desk" can only be told apart by the
**prompt** — and full FT erodes exactly the language conditioning that has to
make that call. Preserving the base (fix 2) + paraphrase aug (below) together
protect this.

Add paraphrase augmentation: keep several rephrasings per task and sample one
per example
(e.g. "put the bottle on the box", "place the bottle onto the box", "pick up
the bottle and set it on the box"). Cleanest insertion point is a small
prompt-rewrite transform in `LeRobotFarmDataConfig.create` (prepend to the
`repack_transform` group) — verify openpi's transform API on the cluster before
wiring it into the patch, since a broken data transform crashes training.

### 4 — Data diversity is the real ceiling
2 tasks, both bottle↔box↔desk. **No architecture recovers broad task
generality from this** — π0.5's generality came from co-training on a large,
diverse mixture. To actually "do most tasks", collect more objects and verbs
(the teleop + review pipeline in `farm serve` already supports this), or
co-train with an open VLA mixture. Set expectations accordingly: the realistic
near-term win is "the trained tasks robustly + modest generalization to bottle
variations and rephrasings", not open-vocabulary manipulation.

### 5 — Jitter is downstream of model uncertainty + chunk seams
Because the data is smooth, jitter comes from (a) flow-matching sample noise
that grows when the policy is uncertain/OOD — fixes 1–4 reduce this directly —
and (b) discontinuity between independent 10-step chunks (RTC is off by
default). The execution path (1€ filter + ω=30 PD tracker) already smooths a
lot. If residual seam-jitter remains after fixing the policy, re-enable **RTC
alone** (not stacked with the 15 Hz/100 Hz streaming experiments, which is what
regressed task performance before):

```bash
# dashboard Run body or CLI:
{"rtc": true}                 # native 30 Hz, all 10 actions, RTC seam-blend on
```

Validate RTC on **uncertain** obs (it has little to fix on confident
in-distribution frames). See `project_rtc_smooth_motion` notes.

## Suggested experiment order
1. Eval checkpoints 5k/10k/15k/19999 → pick best on held-out + novel positions. *(no GPU-train)*
2. `sbatch train_pi05_lora.sbatch` → A/B LoRA vs the chosen full-FT checkpoint.
3. If language is the gap, add prompt-paraphrase aug and retrain the winner.
4. Collect more diverse demos — this is the lever for true generality.
5. Only after the policy is solid: re-enable tuned RTC for the last bit of smoothness.
