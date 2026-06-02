# Inference & deployment — smooth, accurate motion (PI-style)

How to run the overnight flagship on the arm so the motion is smooth and
accurate. Companion to `model/cluster/DEPLOYMENT.md` (port-forward details) and
`farm_pi05_domain_robustness.pdf` (why this model).

## The action/inference contract (verified)

The flagship is GSE (`pi05_farm_uf850_gse`), `action_horizon=10`, **absolute**
joint targets — the *same* contract as the vanilla GSE that was already served,
so `eval_pi05.py` and `serve_pi05.sbatch` work unchanged. The eval client
defaults (`--action-mode absolute`, `resize_with_pad` to 224, base+wrist, 30 Hz)
match training. Single-chunk accuracy is validated by the offline eval
(`create_trained_policy` applies the exact serving transform stack).

## Smoothness — RTC (Real-Time Chunking), PI's technique

Independent action chunks can disagree at their seam → a visible jerk every
~10 steps. RTC inpaints each new chunk to **join the previous one over their
overlap** (async, latency-hiding) — this is how PI gets smooth real-time motion.

- **Server**: the RTC patch is applied (`patch_openpi_rtc.py`; verified in
  `pi0.py` + `policy.py`). Re-served checkpoints pick it up automatically.
- **Client**: `eval_pi05.py` sends RTC fields and **RTC is ON by default**
  (`--no-rtc` disables). Validated on the flagship (`rtc_check.sbatch`, job 155):
  guidance cuts chunk-to-chunk deviation **79%** in the sampler and **47%** on
  the served `Policy.infer` path, while leaving confident in-distribution actions
  essentially unchanged (RTC only acts where chunks disagree).
- The execution path already smooths further: a 1€ filter + a 250 Hz PD tracker
  on the daemon side.
- **Do NOT** also enable `--stream-hz` (100 Hz interpolation). Stacking it on RTC
  regressed task performance before. Add ONE smoothness change at a time.

## Serve the model

```bash
# DEFAULT = the domain-robust flagship (best when the deploy room ≠ training room)
sbatch model/cluster/serve_pi05.sbatch
#   → pi05_farm_uf850_gse · NoahWeiss/farm_uf850_pi05_gse_robust · step-2999

# Alternative: the original full fine-tune (tighter fit IN the training room)
SERVE_CONFIG=pi05_farm_uf850 SERVE_REPO=NoahWeiss/farm_uf850_pi05 \
  SERVE_STEP=19999 sbatch model/cluster/serve_pi05.sbatch
```

Then port-forward worker→login→laptop (see DEPLOYMENT.md).

## Morning on-arm protocol

1. **Compare environments first** (the lesson that diagnosed the live failure):
   `curl -s http://127.0.0.1:8787/v1/cameras/base.jpg -o /tmp/live.jpg` and eyeball
   it against a training frame (`analysis/overnight/sample_base.jpg`). If the
   room/camera differ a lot, you're in the domain-shift case → use the flagship.
2. **Dry-run** (no motion): `python model/eval_pi05.py --task "<task>" --dry-run`
   — confirm chunks infer and action magnitudes look sane.
3. **Live, RTC on** (default): set `drive_real_arm=true` on the dashboard, then
   `python model/eval_pi05.py --task "<task>" --live`. Watch the seams.
4. **A/B RTC**: rerun with `--no-rtc` and compare smoothness. Keep whichever is
   smoother on the arm (RTC expected smoother). If RTC ever hurts task success,
   `--no-rtc` is the proven fallback.

## Which model? (honest)

For a **different room than training** (the actual failure) → **flagship**: it
barely degrades under a room-change (1.33°→1.31° on the stacked proxy) where the
full FT degrades 0.73°→1.70°. For the **original training room**, the full FT
fits tighter (0.73° clean). Domain randomization narrows the gap but doesn't
replace in-domain data — the durable fix is a handful of demos in the target room.
```
