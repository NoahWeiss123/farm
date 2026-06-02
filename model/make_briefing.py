#!/usr/bin/env python3
"""Generate the project BRIEFING PDF — a plain-language explainer of the
overnight π0.5 domain-robustness run: the goal, why the deployed model fails,
why we chose this approach, exactly what is training, and how we judge it.

This is the "what & why" document. The "results" live in the separate
make_report.py PDF.

  python model/make_briefing.py --outdir analysis/overnight \
      --gallery analysis/overnight/fig_perturbation_gallery.png \
      --pdf farm_pi05_briefing.pdf
"""
from __future__ import annotations

import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

INK = "#111827"
MUT = "#6b7280"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11})


def _blank():
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax


def prose(title, blocks, page_no=None):
    """A text page. `blocks` is a list of (kind, text): kind in {h, p, b}
    (header / paragraph / bullet)."""
    fig, ax = _blank()
    ax.text(0.07, 0.93, title, fontsize=21, fontweight="bold", color=INK, va="top")
    ax.plot([0.07, 0.93], [0.895, 0.895], color="#2563eb", lw=2.5)
    y = 0.85
    for kind, text in blocks:
        if kind == "h":
            y -= 0.012
            ax.text(0.07, y, text, fontsize=13.5, fontweight="bold", color="#2563eb", va="top")
            y -= 0.044
        elif kind == "b":
            wrapped = textwrap.fill(text, width=104)
            lines = wrapped.split("\n")
            ax.text(0.085, y, "•", fontsize=11, color="#2563eb", va="top")
            ax.text(0.105, y, lines[0], fontsize=11, color=INK, va="top", family="monospace")
            for ln in lines[1:]:
                y -= 0.030
                ax.text(0.105, y, ln, fontsize=11, color=INK, va="top", family="monospace")
            y -= 0.040
        else:  # paragraph
            wrapped = textwrap.fill(text, width=108)
            ax.text(0.07, y, wrapped, fontsize=11, color=INK, va="top", family="monospace", linespacing=1.45)
            y -= 0.034 * (wrapped.count("\n") + 1) + 0.018
    if page_no:
        ax.text(0.93, 0.04, page_no, fontsize=9, color=MUT, ha="right")
    return fig


def cover():
    fig, ax = _blank()
    ax.add_patch(Rectangle((0, 0.62), 1, 0.38, color="#1e3a8a"))
    ax.text(0.5, 0.83, "FARM · π0.5", fontsize=15, color="#93c5fd", ha="center", fontweight="bold")
    ax.text(0.5, 0.745, "Training a Domain-Robust", fontsize=33, color="white", ha="center", fontweight="bold")
    ax.text(0.5, 0.675, "Manipulation Policy", fontsize=33, color="white", ha="center", fontweight="bold")
    ax.text(0.5, 0.55, "Project briefing — the goal, the diagnosis, and exactly what's running",
            fontsize=13.5, color=INK, ha="center")
    ax.text(0.5, 0.50, "UF850 arm  ·  bottle pick-and-place  ·  overnight 4× H100 run",
            fontsize=11.5, color=MUT, ha="center")
    # one-line thesis
    ax.add_patch(FancyBboxPatch((0.12, 0.30), 0.76, 0.13, boxstyle="round,pad=0.02",
                                fc="#eff6ff", ec="#2563eb", lw=1.5))
    ax.text(0.5, 0.365,
            "The deployed policy fails live not because the model is weak, but because the\n"
            "deploy ROOM differs from the training room. You can't fix a domain shift with a\n"
            "better fine-tune — so we train for robustness to appearance change instead.",
            fontsize=11.5, color=INK, ha="center", va="center", family="monospace", linespacing=1.5)
    ax.text(0.5, 0.16, "Prepared autonomously during the run.  See farm_pi05_domain_robustness.pdf for results.",
            fontsize=10, color=MUT, ha="center")
    return fig


def diagram_pipeline():
    fig, ax = _blank()
    ax.text(0.07, 0.93, "What FARM is — the loop", fontsize=21, fontweight="bold", color=INK, va="top")
    ax.plot([0.07, 0.93], [0.895, 0.895], color="#2563eb", lw=2.5)
    steps = [
        ("VR teleop\nthe arm", "#dbeafe"),
        ("record 200\ndemos", "#dbeafe"),
        ("export to\nLeRobot dataset", "#dcfce7"),
        ("fine-tune π0.5\non H100s", "#fef9c3"),
        ("serve the\ncheckpoint", "#fed7aa"),
        ("drive the\nUF850 arm", "#fecaca"),
    ]
    n = len(steps); x0 = 0.05; w = 0.135; gap = (0.90 - n * w) / (n - 1); y = 0.66; h = 0.13
    cx = []
    for i, (label, c) in enumerate(steps):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=c, ec="#374151", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color=INK)
        cx.append(x + w)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.6))
    ax.text(0.5, 0.60, "π0.5 = a ~3.3B-parameter vision-language-action model (a pretrained generalist robot policy).",
            ha="center", fontsize=10.5, color=MUT, family="monospace")
    body = [
        ("h", "The concrete goal tonight"),
        ("p", "Produce the best deployable policy we can, on 4 GPUs overnight, for the two bottle "
              "tasks ('put the bottle on the box' and 'move it to the desk') — one that actually works "
              "when run on the arm, not just one that scores well offline."),
        ("h", "The data we have"),
        ("p", "200 teleop demos / ~59k frames / 2 tasks, all recorded in ONE environment (a living room). "
              "We are NOT collecting new data tonight — we only have these demos to work with."),
    ]
    y = 0.50
    for kind, text in body:
        if kind == "h":
            ax.text(0.07, y, text, fontsize=13.5, fontweight="bold", color="#2563eb", va="top"); y -= 0.045
        else:
            wrapped = textwrap.fill(text, width=108)
            ax.text(0.07, y, wrapped, fontsize=11, color=INK, va="top", family="monospace", linespacing=1.45)
            y -= 0.034 * (wrapped.count("\n") + 1) + 0.02
    return fig


def diagram_envshift():
    fig, ax = _blank()
    ax.text(0.07, 0.93, "The diagnosis — a domain shift, not a weak model", fontsize=20, fontweight="bold",
            color=INK, va="top")
    ax.plot([0.07, 0.93], [0.895, 0.895], color="#2563eb", lw=2.5)
    # two panels
    ax.add_patch(FancyBboxPatch((0.06, 0.55), 0.40, 0.27, boxstyle="round,pad=0.01", fc="#ecfdf5", ec="#16a34a", lw=1.6))
    ax.text(0.26, 0.79, "TRAINING (what π0.5 saw)", ha="center", fontsize=12, fontweight="bold", color="#15803d")
    for i, t in enumerate(["living-room background", "low frontal camera pose", "consistent lighting",
                           "no people in frame"]):
        ax.text(0.09, 0.74 - i * 0.043, "• " + t, fontsize=10.5, color=INK, family="monospace")
    ax.add_patch(FancyBboxPatch((0.54, 0.55), 0.40, 0.27, boxstyle="round,pad=0.01", fc="#fef2f2", ec="#dc2626", lw=1.6))
    ax.text(0.74, 0.79, "DEPLOY (what it actually sees)", ha="center", fontsize=12, fontweight="bold", color="#b91c1c")
    for i, t in enumerate(["plywood-wall background", "higher / wider camera", "different lighting",
                           "a person standing in frame"]):
        ax.text(0.57, 0.74 - i * 0.043, "• " + t, fontsize=10.5, color=INK, family="monospace")
    ax.annotate("", xy=(0.54, 0.685), xytext=(0.46, 0.685),
                arrowprops=dict(arrowstyle="-|>", color="#374151", lw=2))
    ax.text(0.5, 0.70, "shift", ha="center", fontsize=9, color=MUT)
    body = [
        ("p", "The vision tower was tuned on the living-room look. In a different room it sees "
              "out-of-distribution images and emits canned / garbage motion. This is why the deployed full "
              "fine-tune AND a later GSE model both failed live in the SAME way."),
        ("h", "Why offline evaluation lied"),
        ("p", "Offline eval replayed the ORIGINAL living-room JPEGs, so it never exercised the new camera or "
              "room — it scored sub-1° error and looked perfect while the live policy was effectively blind. "
              "It measured memorization of the training frames, not transfer to the deploy scene."),
        ("h", "Secondary contributors (smaller, but real)"),
        ("b", "Full fine-tuning all 3.3B params on just 2 tasks memorizes two canned trajectories AND "
              "erases π0.5's broad pretrained priors."),
        ("b", "Only 2 fixed prompt strings degrade the language pathway — and the two tasks are "
              "distinguishable ONLY by the prompt."),
        ("p", "PRINCIPLE: no fine-tuning architecture fixes a domain shift. You either change the data "
              "(collect demos in the new room) or make the model robust to appearance change."),
    ]
    y = 0.49
    for kind, text in body:
        if kind == "h":
            ax.text(0.07, y, text, fontsize=13, fontweight="bold", color="#2563eb", va="top"); y -= 0.043
        elif kind == "b":
            wrapped = textwrap.fill(text, width=100)
            lines = wrapped.split("\n")
            ax.text(0.085, y, "•", fontsize=11, color="#2563eb", va="top")
            ax.text(0.105, y, lines[0], fontsize=10.5, color=INK, va="top", family="monospace")
            for ln in lines[1:]:
                y -= 0.029; ax.text(0.105, y, ln, fontsize=10.5, color=INK, va="top", family="monospace")
            y -= 0.038
        else:
            wrapped = textwrap.fill(text, width=108)
            ax.text(0.07, y, wrapped, fontsize=10.7, color=INK, va="top", family="monospace", linespacing=1.4)
            y -= 0.032 * (wrapped.count("\n") + 1) + 0.016
    return fig


def table_2x2():
    fig, ax = _blank()
    ax.text(0.07, 0.93, "Exactly what we're training — a controlled 2×2", fontsize=20, fontweight="bold",
            color=INK, va="top")
    ax.plot([0.07, 0.93], [0.895, 0.895], color="#2563eb", lw=2.5)
    ax.text(0.07, 0.85, textwrap.fill(
        "All four cells are GSE (base-preserving), same recipe (batch 128, 3000 steps ≈ 6.5 epochs, "
        "4×H100, ~70 min each). Only the two augmentation knobs change — so any difference is "
        "attributable to the knob, not luck. The 'neither' cell is the already-trained vanilla GSE.",
        width=108), fontsize=10.8, color=INK, va="top", family="monospace", linespacing=1.45)
    # grid
    gx, gy, cw, ch = 0.30, 0.40, 0.28, 0.13
    ax.text(gx + cw, 0.70, "prompt-aug OFF", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(gx + 2 * cw, 0.70, "prompt-aug ON", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(gx - 0.01, gy + ch + ch / 2, "default\nvisual aug", ha="right", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(gx - 0.01, gy + ch / 2, "HEAVY domain\nrandomization", ha="right", va="center", fontsize=11, fontweight="bold", color=INK)
    cells = {
        (0, 1): ("gse (vanilla)", "already trained", "#f3f4f6"),
        (1, 1): ("gse_prompt", "isolates language", "#fef9c3"),
        (0, 0): ("gse_aug", "isolates vision", "#dbeafe"),
        (1, 0): ("gse_robust  ★", "FLAGSHIP — both", "#dcfce7"),
    }
    # (row, col): row 1=top(default), row0=bottom(heavy); col1=left(off), col0... fix mapping
    layout = {(0, 0): cells[(0, 1)], (0, 1): cells[(1, 1)], (1, 0): cells[(0, 0)], (1, 1): cells[(1, 0)]}
    for (r, c), (name, sub, col) in layout.items():
        x = gx + c * cw; y = gy + (1 - r) * ch
        ax.add_patch(Rectangle((x, y), cw, ch, fc=col, ec="#374151", lw=1.3))
        ax.text(x + cw / 2, y + ch * 0.62, name, ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x + cw / 2, y + ch * 0.28, sub, ha="center", fontsize=9.5, color=MUT)
    body = [
        ("h", "Why GSE (not a plain full fine-tune)"),
        ("p", "GSE = Generalized & Specialized Experts. It SVD-splits each weight into a PRESERVED part "
              "(keeps π0.5's robust pretrained vision/language priors) and an ADAPTED part (learns our task). "
              "A full fine-tune instead overwrites those priors — which is part of why it failed live."),
        ("p", "We also keep the deployed full fine-tune (20k steps) in the comparison as the reference baseline."),
    ]
    y = 0.30
    for kind, text in body:
        if kind == "h":
            ax.text(0.07, y, text, fontsize=13, fontweight="bold", color="#2563eb", va="top"); y -= 0.043
        else:
            wrapped = textwrap.fill(text, width=108)
            ax.text(0.07, y, wrapped, fontsize=10.7, color=INK, va="top", family="monospace", linespacing=1.4)
            y -= 0.032 * (wrapped.count("\n") + 1) + 0.016
    return fig


def image_page(png, title, caption):
    if not png or not os.path.exists(png):
        return None
    fig, ax = _blank()
    ax.text(0.07, 0.95, title, fontsize=19, fontweight="bold", color=INK, va="top")
    ax.plot([0.07, 0.93], [0.915, 0.915], color="#2563eb", lw=2.5)
    img = imread(png)
    iax = fig.add_axes([0.06, 0.30, 0.88, 0.56]); iax.axis("off"); iax.imshow(img)
    ax.text(0.5, 0.24, textwrap.fill(caption, width=110), ha="center", va="top", fontsize=10.8,
            color=INK, family="monospace", linespacing=1.45)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gallery", default="")
    ap.add_argument("--pdf", default="farm_pi05_briefing.pdf")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pages = [
        cover(),
        diagram_pipeline(),
        prose("The problem we're solving", [
            ("h", "The symptom"),
            ("p", "The currently-deployed full fine-tune gets ~30% success on the TRAINED bottle tasks, "
                  "fails completely outside them, and moves jerkily."),
            ("p", "The decisive clue: offline evaluation scored under 1 degree of action error and looked "
                  "great — yet the live policy on the arm was effectively blind. A model that 'looks "
                  "perfect' offline but fails live is almost never a capacity problem; it's an INPUT problem "
                  "— at run time it is seeing something different from what it trained on."),
            ("h", "So the question becomes"),
            ("p", "What is different between the frames the policy trained on and the frames it sees live? "
                  "(Answer on the next page.) Everything we do tonight follows from that answer."),
        ], page_no="3"),
        diagram_envshift(),
        prose("The strategy — why we're doing it this way", [
            ("p", "We cannot collect demos in the deploy room tonight (no physical access). So we use the "
                  "two training-time levers that attack the diagnosed causes using ONLY the data we have:"),
            ("h", "1.  Visual domain randomization  (the main lever)"),
            ("p", "Aggressively perturb the training images every step — brightness, contrast, saturation, "
                  "HUE, per-channel gamma, occasional grayscale, blur, and stronger crop/rotate. This forces "
                  "the vision encoder to key on the TASK (where is the bottle, where is the box) rather than "
                  "the living-room look. A new room then looks like just one more perturbation it has "
                  "already learned to ignore. This is the standard sim-to-real / cross-environment mitigation."),
            ("h", "2.  Base preservation via GSE"),
            ("p", "Fine-tune in a way that KEEPS π0.5's pretrained vision/language priors (which are "
                  "inherently robust to scene changes), instead of overwriting them like a full fine-tune."),
            ("h", "3.  Prompt paraphrasing"),
            ("p", "Sample varied phrasings of each task ('put the bottle on the box', 'place it onto the "
                  "box', ...) so the language pathway stays healthy and the two tasks — which are ONLY "
                  "distinguishable by prompt — don't blur together."),
        ], page_no="5"),
        table_2x2(),
        prose("How we judge it — clean fit vs. domain-shift robustness", [
            ("p", "Both scores are open-loop (no robot): each model replays the 6 held-out episodes and its "
                  "predicted action is compared to the recorded ground truth."),
            ("h", "Clean fit  (the misleading metric)"),
            ("p", "Error on the ORIGINAL frames. This is what looked perfect before while the live policy "
                  "failed — it mostly measures memorization, not transfer. We report it, but it is not the "
                  "metric that matters."),
            ("h", "Domain-shift robustness  (the headline)"),
            ("p", "We PERTURB each frame (see the gallery, next page) — dark, bright, hue-shift, blur, "
                  "occlusion, a stacked 'room-change combo' — BEFORE asking the policy to act, and measure "
                  "how much the action degrades. This is a proxy for the real room change. Perturbations are "
                  "seeded so every model sees byte-identical inputs (a fair comparison)."),
            ("h", "⚠  The honest caveat"),
            ("p", "Synthetic perturbations are a PROXY, not the real plywood room. A model that holds up here "
                  "is more LIKELY to transfer, but the only true validation is live deployment / collecting a "
                  "few demos in the target room. We are explicit about this so the numbers aren't oversold."),
        ], page_no="7"),
        image_page(args.gallery, "What the robustness eval actually tests",
                   "One real training frame (base camera) shown under each perturbation condition. The eval "
                   "asks: when the scene looks like this instead of the clean original, how much does each "
                   "model's predicted action drift? The 'room-change combo' (bottom-right) stacks several to "
                   "mimic a genuine room change."),
        prose("What is running right now", [
            ("h", "One SLURM job on 4× H100"),
            ("b", "Builds the container once, computes normalization stats, then trains the three GSE "
                  "variants back-to-back (3000 steps each, ~70 min each), streaming checkpoints to a "
                  "HuggingFace repo per variant."),
            ("b", "Flagship (gse_robust) is already done: loss fell 0.0744 -> 0.0019, matching vanilla GSE's "
                  "fit despite the heavy augmentation (the augmentation was 'free' — no loss of task fit)."),
            ("h", "The engineering behind it"),
            ("b", "patch_openpi_aug.py — makes openpi's image augmentation env-controlled and adds the heavy "
                  "domain-randomization recipe (backward-compatible; the serving path is unchanged)."),
            ("b", "farm_prompt_aug.py — env-gated prompt paraphrasing (identity at serve time)."),
            ("b", "eval_robust.py — the domain-shift robustness eval.  make_report.py — the results PDF."),
            ("b", "We caught and fixed a NaN-loss bug (gamma applied to negative pixels) in a 20-step smoke "
                  "test BEFORE committing any of the 4-GPU budget."),
            ("h", "Budget & sequence"),
            ("b", "4 GPUs at a time, ~20 of the 40 GPU-hours. Sequence: train (~4.5h) -> one 1-GPU eval pass "
                  "over all 5 models (clean + robust) -> final report PDF in this folder."),
        ], page_no="9"),
        prose("What to expect, and the honest limits", [
            ("h", "The hypothesis we're testing"),
            ("p", "On clean frames every model looks fine. Under the perturbations, the flagship (heavy aug + "
                  "prompt aug, base-preserving) should degrade the LEAST, while the deployed full fine-tune "
                  "collapses — which would both explain the live failure and point to the fix."),
            ("h", "If confirmed"),
            ("p", "Deploy the flagship (NoahWeiss/farm_uf850_pi05_gse_robust, step-2999). The serve command "
                  "is in the results PDF and DEPLOYMENT.md."),
            ("h", "What this does NOT claim"),
            ("p", "Domain randomization NARROWS the train/deploy gap; it does not close it. It cannot "
                  "substitute for in-domain data. The real, durable fix remains: collect a few demos in the "
                  "plywood room (ideally across a couple of lighting conditions) and fine-tune on them."),
            ("h", "The one habit that would have caught this sooner"),
            ("p", "Before blaming the model, compare a LIVE camera frame against a training frame. The whole "
                  "domain-shift diagnosis came from literally looking at the two images side by side."),
        ], page_no="10"),
    ]

    pdf_path = os.path.join(args.outdir, args.pdf)
    with PdfPages(pdf_path) as pdf:
        for fig in pages:
            if fig is None:
                continue
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {pdf_path}  ({sum(1 for f in pages if f)} pages)")


if __name__ == "__main__":
    main()
