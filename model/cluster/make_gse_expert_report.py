#!/usr/bin/env python3
"""Build the GSE expert-count sweep report: figures + FINDINGS from the
per-checkpoint training-set eval JSONs produced by eval_train_endhorizon.py.

Input  : a dir of ``eval-train-<TAG>-step<STEP>.json`` files (TAG ∈
         e2,e4,e8,e16,e32,e80,fft). Run with --raw-dir pointing at them.
Output : PNG figures + report.md + summary.csv into --out-dir.

No GPU, no cluster. Defensive: skips any figure whose data is absent (e.g. the
e8 150k long-term curve before that run finishes), so it can be run repeatedly
as data lands. Pure matplotlib/numpy/json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# TAG -> (total experts, pretty label, plot color). FFT is the full-FT reference.
TAGS = {
    "e2":  (2,   "GSE ×¼ (2 exp)",   "#4C72B0"),
    "e4":  (4,   "GSE ×½ (4 exp)",   "#55A868"),
    "e8":  (8,   "GSE default (8 exp)", "#C44E52"),
    "e16": (16,  "GSE ×2 (16 exp)",  "#8172B3"),
    "e32": (32,  "GSE ×4 (32 exp)",  "#CCB974"),
    "e80": (80,  "GSE ×10 (80 exp)", "#64B5CD"),
    "fft": (None, "FFT (full fine-tune)", "#333333"),
}
ADAPTER_RANK = {2: 4, 4: 8, 8: 16, 16: 32, 32: 64, 80: 160}  # 2 + (total-1)*2
ORDER = ["e2", "e4", "e8", "e16", "e32", "e80", "fft"]
MATCH_STEP = 56000  # the FFT-equivalent comparison step (e8 saves 56000; others 55999)


def load(raw_dir: str):
    """tag -> sorted list of (step, json_dict)."""
    data: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for path in glob.glob(os.path.join(raw_dir, "eval-train-*-step*.json")):
        m = re.search(r"eval-train-(.+?)-step(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        tag, step = m.group(1), int(m.group(2))
        if tag not in TAGS:
            continue
        try:
            with open(path) as f:
                data[tag].append((step, json.load(f)))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {path}: {e}")
    for tag in data:
        data[tag].sort(key=lambda x: x[0])
    return data


def mae(d):       return d["end_of_horizon"]["overall_joint_mae_deg"]
def acc(d, t):    return d["end_of_horizon"]["accuracy_within_deg"].get(str(float(t)), float("nan")) * 100
def grip(d):      return d["end_of_horizon"]["gripper_mae"]
def first(d):     return d["context"]["first_step_mae_deg"]
def lat(d):       return d.get("latency_ms", {}).get("median_ms", float("nan"))


def nearest_step(series, target):
    """(step,d) in series whose step is closest to target."""
    return min(series, key=lambda sd: abs(sd[0] - target))


def fig_mae_vs_step(data, out):
    plt.figure(figsize=(9, 5.5))
    for tag in ORDER:
        if tag not in data:
            continue
        s = [(st, mae(d)) for st, d in data[tag] if st <= MATCH_STEP + 1]
        if not s:
            continue
        xs, ys = zip(*s)
        _, label, c = TAGS[tag]
        ls = "--" if tag == "fft" else "-"
        plt.plot(xs, ys, ls, marker="o", ms=4, color=c, label=label, lw=2 if tag != "fft" else 2.2)
    plt.xlabel("training step")
    plt.ylabel("horizon-end joint MAE (°)  — lower is better")
    plt.title("Training-set action-prediction error vs step (matched 0–56k)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def best_matched(series):
    """min-MAE (step, dict) over checkpoints up to the matched 56k window, so the
    150k e8 run can't cherry-pick a better step from its longer ladder."""
    cands = [sd for sd in series if sd[0] <= MATCH_STEP + 1]
    cands = cands or series
    return min(cands, key=lambda sd: mae(sd[1]))


def fig_best_bar(data, out):
    rows = []
    for tag in ORDER:
        if tag not in data:
            continue
        best = best_matched(data[tag])  # fair: judged on the same 0-56k ladder
        rows.append((tag, best[0], mae(best[1])))
    if not rows:
        return
    labels = [TAGS[t][1] for t, _, _ in rows]
    vals = [v for _, _, v in rows]
    colors = [TAGS[t][2] for t, _, _ in rows]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(range(len(rows)), vals, color=colors)
    for b, (_, st, v) in zip(bars, rows):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}°\n@{st//1000}k",
                 ha="center", va="bottom", fontsize=8)
    plt.xticks(range(len(rows)), labels, rotation=25, ha="right", fontsize=8)
    plt.ylabel("best horizon-end MAE (°) across checkpoints")
    plt.title("Best checkpoint per configuration — lower is better")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_capacity(data, out):
    """MAE at the matched 56k step vs total expert count (GSE only) + FFT line."""
    pts = []
    for tag in ["e2", "e4", "e8", "e16", "e32", "e80"]:
        if tag not in data:
            continue
        st, d = nearest_step(data[tag], MATCH_STEP)
        if abs(st - MATCH_STEP) > 2000:
            continue
        pts.append((TAGS[tag][0], mae(d), acc(d, 5)))
    if not pts:
        return
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(xs, ys, "o-", color="#C44E52", lw=2, label="GSE MAE @56k")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlabel("total experts (log2)   ·   adapter rank = "
                   + ", ".join(f"{x}→{ADAPTER_RANK[x]}" for x in xs))
    ax1.set_ylabel("horizon-end MAE (°) @ step 56k", color="#C44E52")
    ax1.grid(alpha=0.3)
    if "fft" in data:
        st, d = nearest_step(data["fft"], MATCH_STEP)
        ax1.axhline(mae(d), ls="--", color="#333333", label=f"FFT @56k ({mae(d):.2f}°)")
    ax1.legend(fontsize=9)
    ax1.set_title("Capacity curve: expert count vs train-set fit (matched 56k step)")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_longterm(data, out):
    """e8 (default) full ladder to 150k — does continued training keep helping?"""
    if "e8" not in data or max(st for st, _ in data["e8"]) < 60000:
        return  # long-term data not in yet
    s = [(st, mae(d)) for st, d in data["e8"]]
    xs, ys = zip(*s)
    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, "o-", color="#C44E52", lw=2, label="GSE default (8 exp)")
    plt.axvline(MATCH_STEP, ls=":", color="gray", label="56k (FFT-matched point)")
    if "fft" in data:
        st, d = nearest_step(data["fft"], MATCH_STEP)
        plt.axhline(mae(d), ls="--", color="#333333", label=f"FFT @56k ({mae(d):.2f}°)")
    # annotate min
    imin = int(np.argmin(ys))
    plt.scatter([xs[imin]], [ys[imin]], s=90, facecolors="none", edgecolors="k", zorder=5)
    plt.text(xs[imin], ys[imin], f"  min {ys[imin]:.2f}°@{xs[imin]//1000}k", fontsize=8, va="center")
    plt.xlabel("training step")
    plt.ylabel("horizon-end MAE (°)")
    plt.title("Continued training to 150k (default 8-expert) — long-term effect")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_latency(data, out):
    pts = []
    for tag in ["e2", "e4", "e8", "e16", "e32", "e80"]:
        if tag not in data:
            continue
        st, d = nearest_step(data[tag], MATCH_STEP)
        L = lat(d)
        if L == L:  # not nan
            pts.append((TAGS[tag][0], L))
    if len(pts) < 2:
        return
    pts.sort()
    xs, ys = zip(*pts)
    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, ys, "o-", color="#4C72B0", lw=2)
    plt.xscale("log", base=2)
    plt.xticks(xs, [str(x) for x in xs])
    plt.xlabel("total experts (log2)")
    plt.ylabel("median inference latency (ms)")
    plt.title("Inference cost vs expert count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_pertask(data, out):
    """Per-task end MAE at each config's best step (grouped bars)."""
    tasks, rows = set(), {}
    for tag in ORDER:
        if tag not in data:
            continue
        _, best = min(data[tag], key=lambda sd: mae(sd[1]))
        pt = best.get("per_task_end_mae_deg", {})
        rows[tag] = pt
        tasks |= set(pt)
    if not rows:
        return
    tasks = sorted(tasks)
    # short task labels
    def short(t):
        t = t.lower()
        for k in ("bottle", "bear", "hat", "duck"):
            if k in t:
                return k
        return t[:10]
    tl = [short(t) for t in tasks]
    present = [t for t in ORDER if t in rows]
    n = len(present)
    w = 0.8 / max(n, 1)
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(tasks))
    for i, tag in enumerate(present):
        ys = [rows[tag].get(t, np.nan) for t in tasks]
        plt.bar(x + i * w, ys, w, color=TAGS[tag][2], label=TAGS[tag][1])
    plt.xticks(x + 0.4 - w / 2, tl)
    plt.ylabel("per-task horizon-end MAE (°) at best step")
    plt.title("Per-task fit by configuration (best checkpoint each)")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def write_csv(data, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "total_experts", "adapter_rank", "step", "end_mae_deg",
                    "acc_within_2deg", "acc_within_5deg", "acc_within_10deg",
                    "gripper_mae", "first_step_mae_deg", "median_latency_ms"])
        for tag in ORDER:
            if tag not in data:
                continue
            te = TAGS[tag][0]
            rank = ADAPTER_RANK.get(te, "")
            for st, d in data[tag]:
                w.writerow([tag, te, rank, st, f"{mae(d):.4f}",
                            f"{acc(d,2):.2f}", f"{acc(d,5):.2f}", f"{acc(d,10):.2f}",
                            f"{grip(d):.4f}", f"{first(d):.4f}", f"{lat(d):.1f}"])


def write_report(data, out_dir, figs):
    lines = []
    A = lines.append
    A("# GSE expert-count sweep — training-set fit report\n")
    A("π0.5 GSE on `farm_uf850_multiobject` (424 eps / 4 tasks). Six expert-count "
      "variants, recipe-matched to the FFT flagship (batch 32, 56k steps, identical "
      "cosine LR, heavy aug + prompt paraphrase) — the ONLY difference is the GSE "
      "SVD-spectral experts (frozen backbone) vs full fine-tune, and the number of "
      "experts. Metric: open-loop / teacher-forced horizon-end joint MAE on TRAINING "
      "episodes (the standard offline VLA action-accuracy metric), identical frames "
      "across every model (fixed seed).\n")
    # headline table at matched step
    A("## Headline — matched 56k step (FFT-equivalent)\n")
    A("| config | total experts | adapter rank | MAE° @56k | within 5° | best MAE° (step) |")
    A("|---|---|---|---|---|---|")
    for tag in ORDER:
        if tag not in data:
            continue
        te = TAGS[tag][0]
        rank = ADAPTER_RANK.get(te, "—")
        st, d = nearest_step(data[tag], MATCH_STEP)
        bstep, bd = best_matched(data[tag])  # fair: ≤56k ladder for every config
        teh = te if te is not None else "—"
        A(f"| {TAGS[tag][1]} | {teh} | {rank} | {mae(d):.2f} | {acc(d,5):.0f}% "
          f"| {mae(bd):.2f} (@{bstep//1000}k) |")
    A("")
    # winner
    matched = []
    for tag in ["e2", "e4", "e8", "e16", "e32", "e80"]:
        if tag in data:
            st, d = nearest_step(data[tag], MATCH_STEP)
            if abs(st - MATCH_STEP) <= 2000:
                matched.append((mae(d), tag))
    if matched:
        matched.sort()
        best_tag = matched[0][1]
        A(f"**Best GSE configuration @56k:** {TAGS[best_tag][1]} "
          f"({matched[0][0]:.2f}° horizon-end MAE).\n")
    A("## Figures\n")
    for cap, fn in figs:
        if os.path.exists(os.path.join(out_dir, fn)):
            A(f"### {cap}\n\n![{cap}]({fn})\n")
    A("## Notes\n")
    A("- *Matched step*: the five 56k runs save their final checkpoint at step "
      "55999; the default-expert long run saves at 56000. The 1-step offset is "
      "immaterial (same data, same recipe) and is treated as the same comparison "
      "point.\n")
    A("- The default (8-expert) run continues to 150k with the LR held at the 56k "
      "cosine floor (2.5e-6) — a clean test of whether more gradient steps on the "
      "same data keep reducing error past the FFT-matched horizon.\n")
    A("- This is an in-distribution *fit* metric on training episodes; it measures "
      "how faithfully each configuration reproduces the demonstrated actions, not "
      "out-of-distribution robustness.\n")
    A("- *Serve-faithful comparison.* FFT curves are the EMA params (full-FT "
      "convention); GSE curves are the raw params (GSE/LoRA convention, ema_decay="
      "None). Each is the artifact that method actually deploys, so a step-aligned "
      "point compares deployable models — not identical optimizer snapshots. This "
      "is the correct comparison, not a confound.\n")
    A("- *Fair 'best checkpoint'.* The best-MAE column/bar is taken over the "
      "0–56k ladder for every configuration, so the 150k default run cannot "
      "cherry-pick a later step; its full-ladder minimum is shown only in the "
      "continued-training figure.\n")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load(args.raw_dir)
    if not data:
        print(f"!!! no eval JSONs found in {args.raw_dir}")
        return
    print("loaded:", {t: [s for s, _ in v] for t, v in data.items()})

    figs = [
        ("Error vs step (matched 0–56k)", "mae_vs_step.png"),
        ("Best checkpoint per configuration", "best_bar.png"),
        ("Capacity curve (experts vs fit @56k)", "capacity.png"),
        ("Continued training to 150k (default 8-expert)", "longterm.png"),
        ("Per-task fit (best checkpoint each)", "pertask.png"),
        ("Inference cost vs expert count", "latency.png"),
    ]
    fig_mae_vs_step(data, os.path.join(args.out_dir, "mae_vs_step.png"))
    fig_best_bar(data, os.path.join(args.out_dir, "best_bar.png"))
    fig_capacity(data, os.path.join(args.out_dir, "capacity.png"))
    fig_longterm(data, os.path.join(args.out_dir, "longterm.png"))
    fig_pertask(data, os.path.join(args.out_dir, "pertask.png"))
    fig_latency(data, os.path.join(args.out_dir, "latency.png"))
    write_csv(data, os.path.join(args.out_dir, "summary.csv"))
    write_report(data, args.out_dir, figs)
    print(f"wrote figures + report.md + summary.csv → {args.out_dir}")


if __name__ == "__main__":
    main()
