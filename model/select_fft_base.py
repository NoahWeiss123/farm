#!/usr/bin/env python3
"""Which FFT checkpoint generalizes best? Combine the two sweeps and recommend the
LoRA base (the `lora_base` symlink), with an honest fit-vs-generalization curve.

Reads from --indir (place all sweep JSONs together):
  * traindata-fft-<step>.json   (eval_fft_ckptgen)  — action accuracy on multi-object
        episodes the FFT trained on → FIT. Tends to improve/plateau with step.
  * eval-clean-fft_<step>.json   (eval_fft_bench fftonly) — clean MAE on the held-out
        eval_bench (separate recordings + an OOD task) → GENERALIZATION.
  * eval-robust-fft_<step>.json  — held-out domain-shift MAE → generalization-under-shift.

WHY both: the FFT trained on ALL 424 multi-object episodes, so training-data accuracy
rewards memorization and favours the final step; the held-out signals reveal whether a
later step over-memorised. We recommend the best HELD-OUT checkpoint (the user's intent:
"best generalization … in case it gets worse later"), and show the training-data curve
beside it so the trade-off is explicit.

  python model/select_fft_base.py --indir analysis/fullFlagshipFFT/fft_sweep
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

DEG = 180.0 / np.pi


def _step(path):
    m = re.search(r"fft[_-](\d+)\.json$", path)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--fig", default="")
    args = ap.parse_args()

    rows: dict[int, dict] = {}
    for p in glob.glob(os.path.join(args.indir, "traindata-fft-*.json")):
        s = _step(p)
        if s is None:
            continue
        d = json.load(open(p))
        rows.setdefault(s, {})["train_deg"] = d["end_of_horizon"]["overall_joint_mae_deg"]
        rows[s]["train_acc5"] = d["end_of_horizon"]["accuracy_within_deg"].get("5.0", float("nan")) * 100
    for p in glob.glob(os.path.join(args.indir, "eval-clean-fft_*.json")):
        s = _step(p)
        if s is None:
            continue
        d = json.load(open(p))
        rows.setdefault(s, {})["held_clean_deg"] = float(d.get("overall_joint_mae_rad", float("nan"))) * DEG
    for p in glob.glob(os.path.join(args.indir, "eval-robust-fft_*.json")):
        s = _step(p)
        if s is None:
            continue
        d = json.load(open(p))
        conds = d.get("conditions", {})
        pert = [v.get("joint_mae_deg") for c, v in conds.items() if c != "clean" and v.get("joint_mae_deg") is not None]
        rows.setdefault(s, {})["held_robust_deg"] = float(np.mean(pert)) if pert else None
    if not rows:
        raise SystemExit(f"no sweep JSONs in {args.indir}")

    steps = sorted(rows)
    cols = [("train_deg", "train°(fit)"), ("train_acc5", "train@5%"),
            ("held_clean_deg", "held°(gen)"), ("held_robust_deg", "heldRobust°")]
    print(f"{'step':>8}  " + "  ".join(f"{h:>12}" for _, h in cols))
    for s in steps:
        r = rows[s]
        print(f"{s:>8}  " + "  ".join(
            f"{(r.get(k) if r.get(k) is not None else float('nan')):>12.3f}" for k, _ in cols))

    def argmin(key):
        cand = {s: rows[s][key] for s in steps if rows[s].get(key) is not None}
        return min(cand, key=cand.get) if cand else None

    best_train = argmin("train_deg")
    best_held = argmin("held_clean_deg")
    best_robust = argmin("held_robust_deg")
    final = steps[-1]
    # recommend the best HELD-OUT generalizer; fall back to robust, then train.
    rec = best_held or best_robust or best_train
    print(f"\nbest FIT (training data):       step {best_train}")
    print(f"best GENERALIZATION (held-out):  step {best_held}")
    print(f"best ROBUST (held-out shift):    step {best_robust}")
    print(f"final step:                      step {final}")
    if best_held is not None and best_held != final:
        verdict = (f"step {best_held} generalizes best and BEATS the final {final} on held-out → "
                   f"use {best_held} as the LoRA base; the final over-memorised (its train fit is "
                   f"better but it generalizes worse) — exactly the failure the user flagged.")
    elif best_held == final:
        verdict = f"the final step {final} also generalizes best — no over-memorisation penalty; use {final}."
    else:
        verdict = f"no held-out data — recommending best training-fit step {best_train} (weaker basis; prefer held-out)."
    print(f">>> RECOMMENDED LoRA base: step {rec}\n    {verdict}")

    out = {"per_step": rows, "best_train": best_train, "best_heldout": best_held,
           "best_robust": best_robust, "final": final, "recommended": rec, "verdict": verdict}
    path = args.out or os.path.join(args.indir, "fft_base_selection.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")
    print(f"\nNext (cluster): ln -sfn {rec} "
          f"$HOME/farm-train/openpi/checkpoints/pi05_farm_multiobject_fft/farm_fft_multiobject_robust_406/lora_base")

    # curve figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for key, lab, col in [("train_deg", "training-data MAE (fit)", "#9ca3af"),
                              ("held_clean_deg", "held-out clean MAE (generalization)", "#16a34a"),
                              ("held_robust_deg", "held-out domain-shift MAE", "#2563eb")]:
            xs = [s for s in steps if rows[s].get(key) is not None]
            ys = [rows[s][key] for s in xs]
            if xs:
                ax.plot(xs, ys, "o-", label=lab, color=col, lw=1.8)
        for s, lab, c in [(best_held, "best gen", "#16a34a"), (final, "final", "#dc2626")]:
            if s is not None:
                ax.axvline(s, ls="--", color=c, alpha=0.5)
        ax.set_xlabel("FFT checkpoint step"); ax.set_ylabel("joint MAE @ horizon end (deg)")
        ax.set_title("FFT checkpoint generalization — fit vs held-out\n(LoRA base = best held-out, guarding the over-memorised final)",
                     fontweight="bold", fontsize=11)
        ax.legend(); ax.grid(alpha=0.3)
        figp = args.fig or os.path.join(args.indir, "fig_fft_ckpt_generalization.png")
        fig.tight_layout(); fig.savefig(figp, dpi=140, bbox_inches="tight")
        print(f"wrote {figp}")
    except Exception as exc:
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
