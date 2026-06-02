# Overnight run — domain-robust π0.5 (FARM UF850)

Trained 2026-05-30 on 4× H100. The deliverable is
**`farm_pi05_domain_robustness.pdf`** (multi-page report). `*_INTERIM.pdf` is a
mid-run snapshot (methodology + flagship loss curve only) and is superseded by
the final PDF.

## Why this run

The deployed full fine-tune fails live mainly because of a **domain shift** —
demos were recorded in one room (living-room background, fixed camera) and the
policy is run in another (plywood wall, repositioned/wider camera, a person in
frame). No fine-tune *architecture* fixes a domain shift. The two training-time
levers that do help, and that we can apply with only the existing data, are:

1. **Visual domain randomization** — aggressively perturb image appearance during
   training (hue / channel-gamma / grayscale / blur + stronger crop-rotate, on top
   of openpi's stock jitter) so the vision encoder stops keying on the exact
   living-room look. Implemented in `model/cluster/patch_openpi_aug.py`
   (env-gated `FARM_AUG_LEVEL=heavy`; backward-compatible).
2. **Prompt paraphrasing** — the dataset has only 2 fixed task strings, so the
   language pathway overfits; the two tasks are *only* distinguishable by prompt.
   `model/cluster/farm_prompt_aug.py` samples a paraphrase per example
   (env-gated `FARM_PROMPT_AUG=1`, identity at serve time).

Both layer on **GSE** (base-preserving: keeps π0.5's robust pretrained vision
priors that the full fine-tune erased).

## The 2×2 ablation

All cells are GSE, same 4-GPU recipe (batch 128, 3k steps ≈ 6.5 epochs), so they
are directly comparable. The "neither" cell is the already-trained vanilla GSE.

| | prompt-aug **off** | prompt-aug **on** |
|---|---|---|
| **default** visual aug | `gse` (vanilla, step-2999) | `gse_prompt` |
| **heavy** domain-randomization | `gse_aug` | **`gse_robust`** ← flagship |

Plus the deployed **`full`** (full-FT step-19999) as the reference baseline.

HF repos: `NoahWeiss/farm_uf850_pi05_gse_robust` / `_gse_aug` / `_gse_prompt`
(step-2999 each); baselines `..._pi05` (full) and `..._pi05_gse` (vanilla).

## How models are scored

Open-loop, no robot. Each model replays the 6 held eval episodes and its predicted
action chunk is compared to the recorded ground truth (`action[t] == state[t+1]`).

- **Clean fit** (`eval_offline.py` → `eval-clean-<model>.json`): error on the
  original frames. Measures memorization, *not* transfer.
- **Domain-shift robustness** (`eval_robust.py` → `eval-robust-<model>.json`): the
  headline. Each frame is perturbed (dark / bright / hue / blur / occlusion /
  "room-change combo" …) before inference — a proxy for the plywood-room shift.
  Perturbations are seeded so every model sees byte-identical inputs.

## ⚠ Honest caveat

Synthetic perturbations are a **proxy** for the real domain shift, not the real
thing. A model that holds up here is *more likely* to transfer, but the only true
validation is live deployment / collecting a few demos in the target room. Read
the robustness numbers as "how gracefully predictions hold up when scene
appearance changes," not as a guaranteed live success rate.

## Files

| file | what |
|---|---|
| `farm_pi05_domain_robustness.pdf` | the report (cover → gallery → robustness → 2×2 → fit → heatmap → per-task → loss → latency → conclusions) |
| `fig_*.png` | individual report figures |
| `eval-clean-*.json` / `eval-robust-*.json` | raw per-frame eval results |
| `sample_base.jpg` | the base-cam frame used for the perturbation gallery |
| `logs/` | training logs (loss curves parsed from here) |

## Results (2026-05-30)

Joint action error (deg), open-loop on 6 held episodes. `Δ` = mean-perturbed −
clean (the **robustness** measure); `room-combo` = the stacked room-change proxy.

| model | clean | mean-perturbed | Δ degrade | room-combo |
|---|---|---|---|---|
| full-FT (deployed) | **0.73** | 1.44 | +0.71 | 1.70 |
| GSE (vanilla) | 1.01 | 1.94 | +0.93 | 2.03 |
| GSE + prompt-aug | 1.05 | 2.07 | +1.03 | 2.18 |
| GSE + heavy-aug | 1.30 | 1.69 | +0.39 | **1.32** |
| **GSE + aug + prompt (flagship)** | 1.33 | 1.71 | **+0.38** | **1.31** |

**Honest read — not a clean sweep:**
- The full FT fits the *original* frames tightest (memorization) and wins on
  *mild* shifts; it even wins on occlusion + heavy noise (perturbation types the
  augmentation didn't include).
- **Heavy visual augmentation buys stability**: smallest degradation under shift
  (Δ +0.38° vs full's +0.71°), and on the realistic stacked **room-change combo**
  the aug models hold ~1.3° while the full FT degrades to 1.70°. The 2×2 isolates
  visual aug as *the* lever (2.03°→1.32°); prompt-aug doesn't help this metric
  (its value is language generalization, not probed here).
- **For a different room than training (the actual live-failure), the flagship is
  the better bet.** For the original room, the full FT is tighter. This is an
  offline proxy — confirm on the arm (`INFERENCE.md`).

RTC (PI-style seam-smoothing) validated on the flagship: cuts chunk-to-chunk
deviation 47–79% (`logs/rtc-check-155.out`). See `INFERENCE.md` to deploy.

GPU cost: ~20.6 H100-hours (≈half the 40h budget); all GPUs released after.

## Regenerate

```bash
# eval (cluster, 1 GPU): scores all 5 models, writes eval-{clean,robust}-*.json
MODELS="full gse gse_robust gse_aug gse_prompt" sbatch model/cluster/eval_all.sbatch
# pull the JSONs + logs to analysis/overnight/, then:
python model/make_report.py --indir analysis/overnight \
    --logs analysis/overnight/logs/train-overnight-*.out \
    --sample-image analysis/overnight/sample_base.jpg \
    --outdir analysis/overnight --pdf farm_pi05_domain_robustness.pdf
```
