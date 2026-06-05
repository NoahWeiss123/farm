#!/usr/bin/env python3
"""Plot horizon-end joint MAE vs training step for the GSE expert sweep + FFT.

One figure, two panels sharing a y-axis:
  left  = "on training episodes"   (per-task head ranges, --split train)
  right = "held-out episodes"      (per-task tail ranges, --split held_out)
Each panel has up to 5 lines: GSE e2/e8/e32/e64 (a light→dark blue ramp so the
expert-count ordering reads at a glance) + FFT (contrasting brick, dashed).

Style matches the house aesthetic in model/make_clean_figures.py:fig_checkpoint
(serif, inward ticks, minor ticks, thin grid, dpi=300).

Reads the JSONs written by eval_curve_inner.sh / eval_train_endhorizon.py:
  <raw-dir>/eval-<split>-<tag>-step<STEP>.json
and plots end_of_horizon.overall_joint_mae_deg. Missing (tag, step) points are
skipped so the figure can be drawn from partial results for previews.

Usage:
  python make_error_vs_step_figure.py --raw-dir ./eval_curve \
      --out ../../analysis/gse_expert_sweep/error_vs_step.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# model tag -> (legend label, colour, linestyle, marker)
# GSE experts as a light→dark blue ramp (¼×→8×); FFT a contrasting brick, dashed.
MODELS = {
    "e2":  ("GSE ×¼  (2 experts)",        "#9ecae1", "-",  "o"),
    "e8":  ("GSE ×1  (8 experts, default)", "#4292c6", "-",  "s"),
    "e32": ("GSE ×4  (32 experts)",       "#2171b5", "-",  "^"),
    "e64": ("GSE ×8  (64 experts)",       "#08306b", "-",  "D"),
    "fft": ("FFT  (full fine-tune)",      "#a4432f", "--", "v"),
}
ORDER = ["e2", "e8", "e32", "e64", "fft"]
SPLITS = [("train", "On training episodes"),
          ("held_out", "Held-out episodes (per-task tails)")]
MAX_STEP = 56001  # cap to the matched 56k window (drops FFT's 64000 tag)

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "axes.labelsize": 12.5, "axes.titlesize": 13, "legend.fontsize": 10,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.55,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
}

_FN = re.compile(r"eval-(train|held_out)-([A-Za-z0-9]+)-step(\d+)\.json$")


def load(raw_dir: str):
    """-> data[split][tag] = sorted list of (step, mae_deg)."""
    data: dict[str, dict[str, list[tuple[int, float]]]] = {
        "train": {}, "held_out": {}}
    for path in glob.glob(os.path.join(raw_dir, "**", "eval-*-step*.json"),
                          recursive=True):
        m = _FN.search(os.path.basename(path))
        if not m:
            continue
        split, tag, step = m.group(1), m.group(2), int(m.group(3))
        if tag not in MODELS or step > MAX_STEP:
            continue
        try:
            with open(path) as f:
                d = json.load(f)
            mae = float(d["end_of_horizon"]["overall_joint_mae_deg"])
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"  !! skip {os.path.basename(path)}: {e}")
            continue
        data[split].setdefault(tag, []).append((step, mae))
    for split in data:
        for tag in data[split]:
            data[split][tag].sort()
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", required=True, help="dir of eval-*.json files")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default="π0.5 action-prediction error vs. training "
                    "step — GSE expert sweep vs. full fine-tune")
    ap.add_argument("--caption", default="")
    args = ap.parse_args()

    data = load(args.raw_dir)
    n_pts = sum(len(v) for s in data.values() for v in s.values())
    if n_pts == 0:
        print(f"!! no usable JSONs under {args.raw_dir}")
        return 1
    print(f">>> loaded {n_pts} (split,tag,step) points from {args.raw_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharey=True)
        for ax, (split, sub) in zip(axes, SPLITS):
            ax.set_axisbelow(True)
            for tag in ORDER:
                pts = data[split].get(tag, [])
                if not pts:
                    continue
                xs = [p[0] / 1000.0 for p in pts]   # steps in 1e3
                ys = [p[1] for p in pts]
                label, color, ls, mk = MODELS[tag]
                ax.plot(xs, ys, ls=ls, color=color, lw=1.7, marker=mk, ms=5.0,
                        mfc=color, mec=color, mew=0.8, label=label, zorder=3)
            ax.set_title(sub)
            ax.set_xlabel(r"Training steps  ($\times10^{3}$)")
            ax.minorticks_on()
            ax.grid(which="minor", visible=False)
            ax.margins(x=0.03)
        axes[0].set_ylabel("Horizon-end joint MAE  (degrees — lower is better)")
        leg = axes[0].legend(loc="upper right", frameon=True, framealpha=1.0,
                             edgecolor="#bcbcbc", borderpad=0.7, handlelength=2.2,
                             labelspacing=0.5, title="model")
        leg.get_frame().set_linewidth(0.7)
        fig.suptitle(args.title, fontsize=13.5, y=0.99)
        if args.caption:
            fig.text(0.5, 0.005, args.caption, ha="center", va="bottom",
                     fontsize=9.5, color="#374151", wrap=True)
        fig.tight_layout(rect=(0, 0.04 if args.caption else 0, 1, 0.96))
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f">>> wrote {args.out}")

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "model", "step", "horizon_end_joint_mae_deg"])
        for split, _ in SPLITS:
            for tag in ORDER:
                for step, mae in data[split].get(tag, []):
                    w.writerow([split, tag, step, f"{mae:.4f}"])
    print(f">>> wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
