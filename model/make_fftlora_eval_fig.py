#!/usr/bin/env python3
"""Base FFT vs FFT+LoRA on held-out episodes → the report's headline eval figure.

Reads the paired cmp-fftbase_<task>.json / cmp-fftlora_<task>.json produced by
eval_fftlora_compare (eval_train_endhorizon, same seed/range → identical frames →
paired). Plots per-task horizon-end joint MAE (base vs +LoRA), the Δ improvement,
and within-5° accuracy, annotated with each task's held-out episode window + n.

  python model/make_fftlora_eval_fig.py --indir analysis/fftLoRA_report/eval \
      --out analysis/fftLoRA_report/fig_fftlora_eval.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["bottle", "bear", "duck", "hat"]
COLOR = {"bottle": "#dc2626", "bear": "#16a34a", "duck": "#eab308", "hat": "#2563eb"}


def load(indir):
    base, lora = {}, {}
    for p in glob.glob(os.path.join(indir, "cmp-fftbase_*.json")):
        d = json.load(open(p)); base[d["model"].replace("fftbase_", "")] = d
    for p in glob.glob(os.path.join(indir, "cmp-fftlora_*.json")):
        d = json.load(open(p)); lora[d["model"].replace("fftlora_", "")] = d
    return base, lora


def mae(d):
    return d["end_of_horizon"]["overall_joint_mae_deg"]


def acc5(d):
    return d["end_of_horizon"]["accuracy_within_deg"].get("5.0", float("nan")) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", default="analysis/fftLoRA_report/fig_fftlora_eval.png")
    args = ap.parse_args()
    base, lora = load(args.indir)
    tasks = [t for t in ORDER if t in base and t in lora] + \
            [t for t in base if t not in ORDER and t in lora]
    if not tasks:
        raise SystemExit(f"no paired cmp-fftbase/fftlora JSONs in {args.indir}")

    bm = [mae(base[t]) for t in tasks]
    lm = [mae(lora[t]) for t in tasks]
    delta = [b - l for b, l in zip(bm, lm)]            # +ve = LoRA improved
    ba = [acc5(base[t]) for t in tasks]
    la = [acc5(lora[t]) for t in tasks]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    x = np.arange(len(tasks)); w = 0.38
    b1 = a1.bar(x - w / 2, bm, w, label="base FFT", color="#9ca3af")
    b2 = a1.bar(x + w / 2, lm, w, label="FFT + task-LoRA", color=[COLOR.get(t, "#444") for t in tasks])
    for bar in list(b1) + list(b2):
        a1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8)
    for i, dl in enumerate(delta):
        a1.annotate(f"Δ{dl:+.2f}°", (i, max(bm[i], lm[i])), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=9, fontweight="bold",
                    color="#16a34a" if dl > 0 else "#b91c1c")
    labels = []
    for t in tasks:
        d = lora[t]
        lab = f"{t}\n[{d.get('ep_range','?')}] n={d.get('n_episodes','?')}"
        if d.get("split") and d["split"] != "held_out":
            lab += f"\n({d['split']})"
        labels.append(lab)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("held-out joint MAE @ horizon end (deg)")
    a1.set_title("Base FFT vs FFT + task-LoRA — held-out action accuracy\n(lower is better; Δ = LoRA's improvement)",
                 fontweight="bold", fontsize=11)
    a1.legend()

    b3 = a2.bar(x - w / 2, ba, w, label="base FFT", color="#9ca3af")
    b4 = a2.bar(x + w / 2, la, w, label="FFT + task-LoRA", color=[COLOR.get(t, "#444") for t in tasks])
    for bar in list(b3) + list(b4):
        a2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.0f}%",
                ha="center", va="bottom", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(tasks, fontsize=9)
    a2.set_ylabel("frames within 5° @ horizon end (%)")
    a2.set_title("Within-5° accuracy (higher is better)", fontweight="bold", fontsize=11)
    a2.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # also emit a compact text table for the report
    print(f"\n{'task':8} {'window':12} {'n':>3} {'base°':>7} {'+LoRA°':>7} {'Δ°':>7} {'baseAcc':>8} {'loraAcc':>8}")
    for i, t in enumerate(tasks):
        print(f"{t:8} {lora[t].get('ep_range','?'):12} {lora[t].get('n_episodes','?')!s:>3} "
              f"{bm[i]:>7.3f} {lm[i]:>7.3f} {delta[i]:>+7.3f} {ba[i]:>7.1f}% {la[i]:>7.1f}%")


if __name__ == "__main__":
    main()
