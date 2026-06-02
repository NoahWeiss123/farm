#!/usr/bin/env python3
"""Assemble the real-episode eval figures + metrics + narrative into one PDF.

Reads metrics.json (from make_analysis.py), the figs/ directory, and (if present)
FINDINGS.md for the narrative. NO data is computed here — pure assembly.
"""
from __future__ import annotations
import json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "farm_gse_realdata_eval.pdf")

# (filename substring, caption) in report order
PAGES = [
    ("compare_indist_vs_ood.png", "Headline: in-distribution fit vs out-of-distribution generalization."),
    ("traj_indist_bottle", "IN-DIST · bottle→box: commanded next-angle (red) vs real demo (black), per joint + gripper."),
    ("traj_indist_stuffed_bear", "IN-DIST · bear→box trajectory overlay."),
    ("traj_indist_rubber_duck", "IN-DIST · duck→box trajectory overlay."),
    ("traj_indist_hat", "IN-DIST · hat→box trajectory overlay."),
    ("disp_indist", "IN-DIST · predicted vs real joint displacement over the 333ms horizon (diagonal = perfect)."),
    ("horizon_indist", "IN-DIST · error growth across the chunk + horizon-end accuracy CDF."),
    ("perjoint_indist", "IN-DIST · per-joint error, motion-direction agreement, velocity match."),
    ("pertask_indist", "IN-DIST · per-task error and accuracy."),
    ("traj_ood", "OOD · trajectory overlay on a held-out episode (predicted next-angle vs real demo)."),
    ("disp_ood", "OOD · predicted vs real displacement on held-out data."),
    ("horizon_ood", "OOD · error growth + accuracy CDF on held-out data."),
    ("perjoint_ood", "OOD · per-joint error, direction, velocity on held-out data."),
    ("pertask_ood", "OOD · per-task error (separates seen-instruction held-out episodes from the never-trained reverse task)."),
]


def cover(pdf, mets):
    fig = plt.figure(figsize=(11, 8.5)); fig.text(0.5, 0.94, "Multiobject GSE π0.5 — Real-Episode Action-Prediction Eval",
                                                  ha="center", fontsize=17, weight="bold")
    fig.text(0.5, 0.905, "open-loop / teacher-forced: real observation in → predicted joint chunk vs the real demonstrated future",
             ha="center", fontsize=10, style="italic", color="#444")
    rows = [["metric", "in-distribution", "out-of-distribution"]]
    def g(t, k, fmt="{:.2f}", scale=1.0):
        v = mets.get(t, {}).get(k)
        return "—" if v is None else fmt.format(v * scale)
    rows += [
        ["frames evaluated", g("indist", "n_samples", "{:.0f}"), g("ood", "n_samples", "{:.0f}")],
        ["first-step MAE (33 ms)", g("indist", "first_mae_deg") + "°", g("ood", "first_mae_deg") + "°"],
        ["horizon-end MAE (333 ms)", g("indist", "end_mae_deg") + "°", g("ood", "end_mae_deg") + "°"],
        ["frames within 2°", g("indist", "acc_within_2", "{:.1f}", 100) + "%", g("ood", "acc_within_2", "{:.1f}", 100) + "%"],
        ["frames within 5°", g("indist", "acc_within_5", "{:.1f}", 100) + "%", g("ood", "acc_within_5", "{:.1f}", 100) + "%"],
        ["displacement r (Pearson)", g("indist", "disp_pearson", "{:.3f}"), g("ood", "disp_pearson", "{:.3f}")],
        ["direction agreement (cosine)", g("indist", "dir_cosine_mean", "{:.3f}"), g("ood", "dir_cosine_mean", "{:.3f}")],
        ["velocity match r", g("indist", "vel_pearson", "{:.3f}"), g("ood", "vel_pearson", "{:.3f}")],
        ["gripper within 0.1", g("indist", "gripper_acc_0.1", "{:.1f}", 100) + "%", g("ood", "gripper_acc_0.1", "{:.1f}", 100) + "%"],
    ]
    ax = fig.add_axes([0.1, 0.4, 0.8, 0.42]); ax.axis("off")
    tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for c in range(3): tbl[(0, c)].set_facecolor("#dfe7f3"); tbl[(0, c)].set_text_props(weight="bold")
    note = ("Model: pi05_farm_multiobject_gse (step-5999, VLA-GSE).  In-dist data = its own training set "
            "NoahWeiss/farm_uf850_multiobject (424 eps, 4 tasks).  OOD data = held-out NoahWeiss/farm_uf850_bottle "
            "(200 eps) — episodes never seen in multiobject training; its 'bottle off box → desk' task is an "
            "instruction the model never trained on.  Metric is single-shot prediction fidelity, not closed-loop "
            "success.  All figures generated from real frames + real inferences (model/cluster/eval_train_endhorizon.py).")
    fig.text(0.1, 0.30, "\n".join(textwrap.wrap(note, 110)), fontsize=8.5, color="#333", va="top")
    pdf.savefig(fig); plt.close(fig)


def narrative(pdf, md_path):
    if not os.path.exists(md_path): return
    txt = open(md_path).read()
    # paginate ~46 lines/page
    lines = []
    for para in txt.splitlines():
        lines += textwrap.wrap(para, 100) or [""]
    per = 46
    for pi in range(0, len(lines), per):
        fig = plt.figure(figsize=(11, 8.5)); fig.text(0.07, 0.95, "Findings" + (" (cont.)" if pi else ""),
                                                      fontsize=13, weight="bold")
        fig.text(0.07, 0.91, "\n".join(lines[pi:pi + per]), fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig); plt.close(fig)


def figpage(pdf, path, caption):
    img = mpimg.imread(path)
    h, w = img.shape[:2]; ar = w / h
    fw = 10.5; fh = min(7.6, fw / ar)
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([(1 - fw / 11) / 2, 0.12, fw / 11, fh / 8.5]); ax.imshow(img); ax.axis("off")
    fig.text(0.5, 0.06, caption, ha="center", fontsize=9, color="#333", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def main():
    mets = json.load(open(os.path.join(HERE, "metrics.json"))) if os.path.exists(os.path.join(HERE, "metrics.json")) else {}
    avail = sorted(os.listdir(FIG)) if os.path.isdir(FIG) else []
    with PdfPages(OUT) as pdf:
        cover(pdf, mets)
        narrative(pdf, os.path.join(HERE, "FINDINGS.md"))
        for sub, cap in PAGES:
            hit = next((f for f in avail if f.startswith(sub) or sub in f), None)
            if hit: figpage(pdf, os.path.join(FIG, hit), cap)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
