# π0.5 FARM policy — performance diagnosis & action plan

> **Update 2026-05-30** — Acted on the domain-shift diagnosis: trained
> domain-randomization variants overnight (heavy image aug ± prompt-aug, on GSE)
> in a controlled 2×2. Honest result: heavy visual aug is the most STABLE under
> appearance shift (best on the realistic "room-change combo": ~1.3° vs the full
> FT's 1.7°, flattest degradation) — but NOT a uniform win (the full FT's tight
> clean fit wins on mild shifts + on occlusion/noise the aug didn't cover). Prompt
> aug didn't help this visual metric. Flagship = `NoahWeiss/farm_uf850_pi05_gse_robust`
> (step-2999); serve_pi05.sbatch now defaults to it. Full report, briefing, and the
> inference/RTC runbook are in `analysis/overnight/`. The durable fix remains
> in-room demos — domain randomization only narrows the gap.

Audit done 2026-05-29 against the deployed full-FT `pi05_farm_uf850`
(step-19999). Symptoms reported: ~30 % success on the **trained** bottle
tasks, **complete failure** outside training, and **jittery** motion.

Everything below was found by static review + local data analysis
(`model/analyze_dataset.py`) — no GPU required. The training-recipe changes
are validated for *syntax* only; they must be confirmed on the cluster.

## TL;DR

| Lever | Effort | Expected impact |
|---|---|---|
| **1. Pick an earlier checkpoint** (5k/10k vs 19999) | none (ckpts exist) | ↑ generalization, ↓ canned-motion, ↓ jitter |
| **2. Compare 3 fine-tunes** (full FT / LoRA / **GSE**) | 1 GPU each | preserve the base → ↑↑ OOD generalization, ↓ memorization |
| **3. Prompt/paraphrase augmentation** | data/transform | ↑ language generalization |
| **4. More diverse data** (more objects/verbs) | data collection | the only real fix for *broad* generality |
| **5. Tuned RTC alone for smoothness** | serve flag | ↓ jitter at chunk seams |

**Three fine-tuning architectures are now set up** in `model/cluster/`, all
directly comparable (same base, data, action contract, serve/eval path):
`pi05_farm_uf850` (full FT, the current ~30% model), `pi05_farm_uf850_lora`
(LoRA — preserves the base), and `pi05_farm_uf850_gse` (VLA-GSE — SVD spectral
experts, the principled middle ground; see `openpi_gse.py`). Train each with its
sbatch and compare on held-out + novel positions. See `model/cluster/README.md`.

## What is NOT wrong (ruled out)

Run `python model/analyze_dataset.py --dataset datasets/lerobot/farm_uf850_bottle --raw datasets/dataset3`:

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
(`model/cluster/patch_openpi_config_pi05_lora.py`, registered by `setup.sh`).
Freezes the backbone, trains low-rank adapters → preserves the base. Fits on a
single H100, so it respects the shared-cluster gpu:1 norm. Same serve/eval path
(action_horizon=10, absolute actions) — no client changes.

```bash
# after setup.sh has registered it:
sbatch model/cluster/train_pi05_lora.sbatch        # → NoahWeiss/farm_uf850_pi05_lora
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
1. Eval the existing full-FT checkpoints 5k/10k/15k/19999 → pick best on held-out
   + novel positions. *(no GPU-train; checkpoints already on HF)*
2. Train + compare the three architectures and pick the winner on held-out:
   `sbatch train_pi05_lora.sbatch`, `sbatch train_pi05_gse.sbatch`
   (GSE: run the smoke test below first), vs the chosen full-FT checkpoint.
3. If language is the gap, add prompt-paraphrase aug and retrain the winner.
4. Collect more diverse demos — this is the lever for true generality.
5. Only after the policy is solid: re-enable tuned RTC for the last bit of smoothness.

## GSE: will it train faster *and* better? (research notes)

From the VLA-GSE paper (arXiv:2605.06175) ablation (Table 4, Long suite), what
actually drives GSE's gains — and what our implementation captures:

| Component | Δ if removed | In our impl? |
|---|---|---|
| **SVD initialization** | **−13.2 pts** (the dominant lever) | **✅ yes** — attention (q/k/v) *and* FFN, verified to ~1e-14 |
| Specialized experts | −11.0 | ✅ present (dense-summed; capacity kept) |
| Generalized expert | −6.9 | ✅ always-on |
| Auxiliary (load-balance) loss | −2.1 | ⚠️ omitted (needs routing; minor) |
| Gradient scaling | −2.0 | ⚠️ omitted (minor) |

So our config captures the big levers — SVD-init + both expert types — and skips
the two ~2-pt refinements. The paper's per-token routing decouples specialists
across *diverse* tasks; on **2 near-identical bottle tasks the routing benefit is
marginal**, so it defaults OFF (dense) and the specialists act as added SVD-init
capacity. That is the right trade for this data.

**Why GSE should be *better* here:** it freezes the backbone (so it can't forget
π0.5's pretrained generalization the way the 21-epoch full FT does) *and*
SVD-initializes the adapters to the dominant subspace (so it adapts harder than
random-init LoRA). It is the principled middle between the two failure modes in
"Root causes" above.

**Why GSE trains *faster*:**
- **GPU-hours:** ~6 H100-hours (1 GPU × ~6 h) vs the full FT's ~28 (8 GPUs ×
  ~3.5 h) — roughly **4–5× cheaper**. It frees the 8-GPU node for others.
- **Convergence:** SVD-init starts the adapters at the dominant subspace (the
  forward ≈ the base at step 0), so it needs **fewer steps** than random-init
  LoRA — checkpoints often peak early (select on held-out, don't assume the last).
- **Wall-clock:** GSE freezes the backbone so it needs no FSDP — it
  data-parallel-replicates across whatever GPUs you give it. `--gres=gpu:1` is
  ~5–7 h; bump to `gpu:2`/`gpu:4` for a ~2–4× wall-clock speedup, still far
  cheaper than full FT. The 1-GPU run is sized to finish well inside its 12 h
  walltime, with checkpoints every 3k so a partial run is still usable.

**Config:** `gemma_2b_gse` VLM (GSE adapters, SVD-init) + LoRA-PiSSA FFN +
full-FT action expert, frozen backbone; batch 32, 12k steps (≈6.5 epochs), a
gentle 2.5e-5 LR. LoRA uses the **same** 12k budget so the head-to-head is fair.

**Refinements left on the table** (validated on the paper's *diverse* data;
marginal on 2 tasks — try only if GSE underperforms): (1) **decoupled LRs** (GSE
1e-5 / action head 1e-4 via an `optax.multi_transform` patch to openpi's
optimizer — the paper's catastrophic-forgetting guard); (2) **per-token routing**
on q/k/v (the gate broadcast is validated for those; the attention-output einsum
needs separate handling); (3) the load-balance + gradient-scaling terms (~+2 each).

## GSE smoke test (run before the full GSE training run)

The GSE integration (`openpi_gse.py` + `patch_openpi_gse.py`) is validated for
**syntax** (every patched openpi file py-compiles) and for **math** (the SVD
init exactly reconstructs each weight block to ~1e-14, verified against all
Gemma einsum equations; the dense forward equals the base at init). It is **not
yet GPU-tested**. Before committing a multi-hour run, confirm it builds and
takes a few optimization steps on one GPU:

```bash
# on the login pod, after setup.sh registered pi05_farm_uf850_gse:
srun --partition=small --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  --container-mounts="$HOME:$HOME" --container-workdir="$HOME/farm-train/openpi" \
  bash -lc 'export HOME='"$HOME"'; export JAX_PLATFORMS=cuda; pip install -q uv; uv sync --frozen;
    uv run scripts/compute_norm_stats.py --config-name=pi05_farm_uf850_gse;
    uv run python scripts/train.py pi05_farm_uf850_gse --exp-name=gse_smoke \
        --overwrite --no-wandb-enabled --num-train-steps=5'
```

Expect: model builds, `GSESVDWeightLoader` runs the SVD init, 5 steps log a
finite decreasing loss. If a shape/JIT error appears it will be in the GSE
adapter wiring — the most likely spots are the routed path (which defaults OFF;
keep `route=False`) and the FFN adapters. The dense (non-routed) attention path
is the validated core.
