#!/usr/bin/env python3
"""Regenerate ALL fftLoRA_report figures cleanly for a non-expert audience:
consistent per-series colors (so legends are correct), plain labels (no cryptic
run-names), a one-line takeaway on each, and the skill-vector story redesigned to
be legible. Reads only saved artifacts (no GPU/cluster).

  python model/make_clean_figures.py --dir analysis/fftLoRA_report
"""
from __future__ import annotations
import argparse, glob, json, os, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": "#e9edf2", "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 12.5, "axes.titlesize": 14.5, "axes.titleweight": "bold", "axes.labelsize": 12.5,
})
BASE_GRAY = "#9aa3af"; WIN = "#0ea5a3"; WARN = "#dc2626"; CLEAN = "#9aa3af"; SHIFT = "#f59e0b"
TASKCOL = {"bottle": "#dc2626", "bear": "#16a34a", "duck": "#eab308", "hat": "#2563eb"}
# pretty display names for the 6 LoRAs
DISP = {"bottle30": "bottle", "bear30": "bear", "duck30": "duck", "hat30": "hat",
        "bottle100": "bottle\n(2× data)", "bottle30s1": "bottle\n(different\nrandom start)"}


def caption(fig, text):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=10.5, color="#374151", wrap=True)


# ───────────────────────── §5: do skills help? ─────────────────────────
def fig_helps(d, out):
    order = ["bottle", "bear", "duck", "hat"]
    base, lora, ns, win = {}, {}, {}, {}
    for p in glob.glob(os.path.join(d, "eval", "cmp-fftbase_*.json")):
        j = json.load(open(p)); t = j["model"].replace("fftbase_", "")
        base[t] = j["end_of_horizon"]["overall_joint_mae_deg"]; ns[t] = j.get("n_episodes", "?")
    for p in glob.glob(os.path.join(d, "eval", "cmp-fftlora_*.json")):
        j = json.load(open(p)); lora[j["model"].replace("fftlora_", "")] = j["end_of_horizon"]["overall_joint_mae_deg"]
    order = [t for t in order if t in base and t in lora]
    fig, ax = plt.subplots(figsize=(10, 5.8)); x = np.arange(len(order)); w = 0.4
    b1 = ax.bar(x - w/2, [base[t] for t in order], w, label="base model (no skill)", color=BASE_GRAY)
    b2 = ax.bar(x + w/2, [lora[t] for t in order], w, label="base + task skill (LoRA)", color=WIN)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012, f"{bar.get_height():.2f}°", ha="center", va="bottom", fontsize=10)
    for i, t in enumerate(order):
        imp = 100*(base[t]-lora[t])/base[t]
        ax.text(i, max(base[t], lora[t])+0.10, f"{imp:.0f}% better", ha="center", fontsize=11, fontweight="bold", color=WIN)
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(n={ns.get(t,'?')} held-out eps)" for t in order])
    ax.set_ylabel("average error of the predicted move  (degrees — lower is better)")
    ax.set_ylim(0, max([base[t] for t in order])*1.32)
    ax.set_title("Does adding a task 'skill' help?  Yes — on episodes it never trained on")
    ax.legend(loc="upper right", framealpha=0.95)
    caption(fig, "Each task gets its own small add-on ('skill'). Tested on held-out episodes the add-on never saw, it cuts the\n"
                 "robot's move-prediction error by ~30–43%. (1° ≈ very precise; a clock's minute hand moves 6° per minute.)")
    fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ───────────────── headline: does a skill fingerprint its task? ─────────────────
def fig_fingerprint(cos, tasks, out):
    idx = {t: i for i, t in enumerate(tasks)}
    real = [t for t in ["bottle30", "bear30", "duck30", "hat30"] if t in idx]
    cross = np.mean([cos[idx[a], idx[b]] for a, b in itertools.combinations(real, 2)])
    moredata = cos[idx["bottle30"], idx["bottle100"]] if "bottle100" in idx else np.nan
    diffseed = cos[idx["bottle30"], idx["bottle30s1"]] if "bottle30s1" in idx else np.nan
    labels = ["DIFFERENT objects\n(same random start)", "SAME object (bottle)\n2× more training data",
              "SAME object (bottle)\nDIFFERENT random start"]
    vals = [cross, moredata, diffseed]; cols = [BASE_GRAY, "#3b82f6", WARN]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(3), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.annotate("same task — yet\nUNRELATED!", (2, diffseed), xytext=(2, 0.45), ha="center", fontsize=12,
                fontweight="bold", color=WARN, arrowprops=dict(arrowstyle="->", color=WARN, lw=2))
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0); ax.set_ylabel("similarity of the two skill add-ons\n(1 = identical, 0 = unrelated)")
    ax.set_title("Do two skills that do the SAME task look alike?  Surprisingly, no")
    caption(fig, "A skill is a set of numbers added to the model. Train the SAME bottle task twice, changing only the random\n"
                 "starting point → the two come out essentially UNRELATED (0.03). So the raw numbers mostly reflect the random\n"
                 "start, not the task — the task itself adds only a small extra similarity (0.72 vs 0.62).")
    fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ───────────────── where does the task live (VLM vs action) ─────────────────
def fig_where(cv, ca, tasks, out):
    idx = {t: i for i, t in enumerate(tasks)}
    real = [t for t in ["bottle30", "bear30", "duck30", "hat30"] if t in idx]
    def crossmean(C): return np.mean([C[idx[a], idx[b]] for a, b in itertools.combinations(real, 2)])
    vals = [crossmean(cv), crossmean(ca)]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.bar(["Vision / Language part\n(what to grasp)", "Action part\n(the arm motion)"], vals,
                  color=["#7c3aed", "#0891b2"], width=0.5)
    for i, v in enumerate(vals): ax.text(i, v+0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("similarity BETWEEN different objects\n(lower = more task-specific)")
    ax.set_title("Where does 'which object' live in the skill?")
    caption(fig, "Across different objects, the vision/language adapters differ more (0.59 → more object-specific) while the\n"
                 "arm-motion adapters are more shared (0.74). Makes sense: WHAT to grab differs by object; HOW to pick-and-place is similar.")
    fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ───────────────── clean similarity heatmap (full model) ─────────────────
def fig_similarity(cos, tasks, out):
    order = [t for t in ["bottle30", "bottle100", "bear30", "duck30", "hat30", "bottle30s1"] if t in tasks]
    oi = [tasks.index(t) for t in order]; M = cos[np.ix_(oi, oi)]
    lab = [DISP.get(t, t).replace("\n", " ") for t in order]
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(lab, fontsize=10)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i,j] < 0.35 else "#111")
    fig.colorbar(im, ax=ax, fraction=0.046, label="similarity (1 = identical, 0 = unrelated)")
    ax.set_title("How similar is every pair of skill add-ons?")
    caption(fig, "Bright green = alike. The 'different random start' bottle (bottom-right) is unrelated to everything (≈0),\n"
                 "even to the other bottle skills — the giveaway that the raw numbers track the random start, not the task.")
    fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ───────────────── clean 'map' of skills (MDS), legend not text labels ─────────────────
def fig_map(cos, tasks, out):
    n = len(tasks); D = np.clip(1-cos, 0, 2); np.fill_diagonal(D, 0)
    J = np.eye(n)-np.ones((n, n))/n; B = -0.5*J@(D**2)@J
    w, V = np.linalg.eigh(B); XY = V[:, -2:]*np.sqrt(np.clip(w[-2:], 0, None))
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    style = {"bottle30": ("bottle", TASKCOL["bottle"], "o"), "bear30": ("bear", TASKCOL["bear"], "o"),
             "duck30": ("duck", TASKCOL["duck"], "o"), "hat30": ("hat", TASKCOL["hat"], "o"),
             "bottle100": ("bottle — 2× data", TASKCOL["bottle"], "s"),
             "bottle30s1": ("bottle — different random start", "#111111", "X")}
    for t in tasks:
        i = tasks.index(t); lab, c, m = style.get(t, (t, "#666", "o"))
        ax.scatter(XY[i, 0], XY[i, 1], c=c, marker=m, s=320, edgecolor="k", linewidth=1.2, label=lab, zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11, frameon=False, title="skill add-on")
    ax.set_title("A 'map' of the skill add-ons  (closer = more alike)")
    caption(fig, "The 'different random start' copy of bottle (black ✕) sits far off on its own, while the four real tasks barely\n"
                 "separate — because the random starting point dominates the raw numbers more than the task does.")
    fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ───────────────── benchmark: FFT vs others (fixed colors → correct legend) ─────────────────
def fig_benchmark(d, out):
    r = json.load(open(os.path.join(d, "phaseC_compare.json")))
    models = ["FFT-multiobj (full FT)", "GSE-multiobj (robust)", "Full FT (2-task bottle)", "LoRA (100-ep bottle)"]
    short = ["This model\n(full fine-tune)★", "GSE\n(alt. method)", "2-task model\n(bottle only)", "LoRA-only\n(1 task, on\nORIGINAL model)"]
    models = [m for m in models if m in r]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.05, 1]})
    x = np.arange(len(models)); w = 0.4
    clean = [r[m]["clean"] for m in models]; shift = [r[m]["rmean"] for m in models]
    b1 = a1.bar(x-w/2, clean, w, label="normal conditions", color=CLEAN)
    b2 = a1.bar(x+w/2, shift, w, label="under visual disturbance", color=SHIFT)
    for bar in list(b1)+list(b2): a1.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9.5)
    a1.set_yscale("log"); a1.set_xticks(x); a1.set_xticklabels(short, fontsize=10)
    a1.set_ylabel("move-prediction error (degrees, log — lower is better)")
    a1.set_title("Which training method is best?"); a1.legend(loc="upper left", framealpha=0.95)
    tasks = ["bottle", "bear", "duck", "hat"]
    M = np.array([[r[m]["pertask"].get(t, np.nan) for t in tasks] for m in models])
    im = a2.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=np.nanpercentile(M, 88))
    a2.set_xticks(range(4)); a2.set_xticklabels(tasks); a2.set_yticks(range(len(models)))
    a2.set_yticklabels([s.split("\n")[0] for s in short], fontsize=10)
    for i in range(len(models)):
        for j in range(4):
            v = M[i, j]; a2.text(j, i, f"{v:.1f}°", ha="center", va="center", fontsize=10, fontweight="bold",
                                 color="white" if v > 0.55*np.nanpercentile(M, 88) else "#111")
    a2.set_title("Can it do each object?  (green = accurate)")
    fig.colorbar(im, ax=a2, fraction=0.045, label="error (deg)")
    caption(fig, "The full fine-tune (this model) is most accurate — in normal conditions AND under visual disturbance — and, with\n"
                 "GSE, is one of only two trained on all four objects. The two 'bottle-only' methods were trained on bottle alone, so\n"
                 "they fail on bear/hat (never seen). NOTE: this 'LoRA-only' is a STANDALONE method on the ORIGINAL model — a weak\n"
                 "baseline — NOT the task add-ons in Result 3, which build on THIS strong model and DO help.")
    fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_checkpoint(d, out):
    sel = json.load(open(os.path.join(d, "fft_sweep", "fft_base_selection.json")))
    ps = sel["per_step"]; steps = sorted(int(s) for s in ps)
    g = lambda k: [ps[str(s)].get(k) for s in steps]
    x = np.array(steps, dtype=float) / 1000.0
    series = [
        (g("train_deg"),       "#8b9097", "o", "train"),
        (g("held_clean_deg"),  "#0f6d75", "s", "held-out"),
        (g("held_robust_deg"), "#a4432f", "^", "held-out, shifted"),
    ]
    style = {
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.labelsize": 12.5, "axes.titlesize": 13, "legend.fontsize": 10.5,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.55,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4.0, "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.7, 4.4))
        ax.set_axisbelow(True)
        for y, c, mk, lab in series:
            ax.plot(x, y, "-", color=c, lw=1.5, marker=mk, ms=5.0, mfc=c, mec=c,
                    mew=0.8, label=lab, zorder=3)
        ax.set_xlabel(r"Training steps  ($\times10^{3}$)")
        ax.set_ylabel(r"Mean joint error  (degrees)")
        ax.set_title("Error vs. training steps")
        ax.minorticks_on(); ax.grid(which="minor", visible=False)
        ax.set_ylim(0.8, 3.15); ax.margins(x=0.03)
        leg = ax.legend(loc="upper right", frameon=True, framealpha=1.0,
                        edgecolor="#bcbcbc", borderpad=0.7, handlelength=2.0, labelspacing=0.55)
        leg.get_frame().set_linewidth(0.7)
        fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="analysis/fftLoRA_report"); a = ap.parse_args()
    d = a.dir
    arr = np.load(os.path.join(d, "lora_vector_arrays.npz"), allow_pickle=True)
    tasks = list(arr["tasks"]); cos = arr["cosine_full"]; cv = arr["cosine_vlm"]; ca = arr["cosine_action"]
    P = lambda f: os.path.join(d, f)
    fig_helps(d, P("clean_1_skills_help.png"))
    fig_fingerprint(cos, tasks, P("clean_2_fingerprint.png"))
    fig_map(cos, tasks, P("clean_3_skill_map.png"))
    fig_where(cv, ca, tasks, P("clean_4_where.png"))
    fig_similarity(cos, tasks, P("clean_5_similarity.png"))
    fig_benchmark(d, P("clean_6_benchmark.png"))
    fig_checkpoint(d, P("clean_7_checkpoint.png"))
    print("wrote 7 clean figures to", d)


if __name__ == "__main__":
    main()
