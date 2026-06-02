#!/usr/bin/env python3
"""Assemble the FFT-LoRA results into a clear, plain-language multi-page PDF
(text pages + the clean figures), for a non-expert reader.

  python model/make_fftlora_report_pdf.py --dir analysis/fftLoRA_report \
      --pdf analysis/fftLoRA_report/farm_fftlora_report.pdf
"""
from __future__ import annotations
import argparse, os
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({"figure.facecolor": "white"})


def text_page(title, body, subtitle=None):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.07, 0.93, title, fontsize=19, fontweight="bold")
    if subtitle:
        fig.text(0.07, 0.885, subtitle, fontsize=11.5, color="#6b7280")
    fig.text(0.07, 0.85, body, va="top", ha="left", fontsize=11.3, linespacing=1.5, wrap=True)
    return fig


def fig_page(png, header):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, header, ha="center", fontsize=15, fontweight="bold", color="#111827")
    if os.path.isfile(png):
        ax = fig.add_axes([0.04, 0.05, 0.92, 0.86]); ax.axis("off")
        ax.imshow(mpimg.imread(png))
    else:
        fig.text(0.5, 0.5, f"(missing: {os.path.basename(png)})", ha="center")
    return fig


def cover():
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.76, "Can a Robot's Skills Be Reusable Building Blocks?", ha="center", fontsize=21, fontweight="bold")
    fig.text(0.5, 0.71, "Task add-ons (LoRAs) on a fine-tuned π0.5 policy — results, explained simply",
             ha="center", fontsize=13, color="#374151")
    fig.text(0.5, 0.665, "FARM · UF850 robot arm · CS153 · " + datetime.now().strftime("%Y-%m-%d"),
             ha="center", fontsize=11.5, color="#6b7280")
    body = (
        "THE SETUP\n"
        "We have a robot arm and an AI model that watches two cameras and decides how to move\n"
        "the arm. We taught it four pick-and-place jobs (bottle, bear, duck, hat → box). Then we\n"
        "tested whether each job could be a small, swappable 'skill add-on' on one shared model.\n\n"
        "WHAT WE FOUND (one line each)\n"
        "  1. The shared model is strong, and training it longer did NOT make it worse.\n"
        "  2. This way of training beats the alternatives (most accurate; handles all 4 objects).\n"
        "  3. Adding a per-task skill add-on makes the robot ~37% more precise — on new episodes.\n"
        "  4. BUT the add-ons are not tidy, comparable 'skill vectors': their raw numbers are\n"
        "     dominated by the random starting point of training, not the task. (Train the SAME\n"
        "     task twice from different random starts → the two add-ons come out UNRELATED.)\n\n"
        "BOTTOM LINE\n"
        "A skill add-on is a real, useful skill — but you can't compare or combine the add-ons by\n"
        "their raw numbers. A skills library has to compare them by what they DO, or start them\n"
        "all from the same point. The detail that makes this trustworthy: we ran the control that\n"
        "most people skip — and it flipped the conclusion.\n\n"
        "How to read the error numbers: the robot predicts the arm's next angles; we report how\n"
        "far off it is, in degrees (lower = better). 1° is very precise — a clock's minute hand\n"
        "moves 6° per minute. 'Held-out' = recordings the model never saw in training."
    )
    fig.text(0.5, 0.37, body, ha="center", va="center", fontsize=10.2, family="monospace", linespacing=1.5)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/fftLoRA_report")
    ap.add_argument("--pdf", default="analysis/fftLoRA_report/farm_fftlora_report.pdf")
    a = ap.parse_args(); d = a.dir; P = lambda f: os.path.join(d, f)
    pages = [
        cover(),
        fig_page(P("clean_7_checkpoint.png"), "Result 1 — The shared model didn't 'over-memorize'"),
        fig_page(P("clean_6_benchmark.png"), "Result 2 — This training method is the most accurate, and covers all 4 objects"),
        fig_page(P("clean_1_skills_help.png"), "Result 3 — Adding a per-task skill add-on helps (on episodes it never saw)"),
        fig_page(P("clean_2_fingerprint.png"), "Result 4 — The surprise: same task, different random start ⇒ UNRELATED add-ons"),
        fig_page(P("clean_3_skill_map.png"), "Result 4 — A 'map' of the add-ons: the different-start copy flies off alone"),
        fig_page(P("clean_4_where.png"), "Result 4 — Where 'which object' lives: more in the vision/language part"),
        fig_page(P("clean_5_similarity.png"), "Result 4 — Full similarity table (for reference)"),
        text_page("What it all means",
            "FUNCTIONALLY, SKILLS WORK.\n"
            "  • The shared model is strong (Results 1–2) and bolting on a task add-on reliably\n"
            "    sharpens it by ~37% on held-out episodes (Result 3).\n\n"
            "BUT THE ADD-ON'S RAW NUMBERS ARE NOT A CLEAN 'SKILL COORDINATE' (Result 4).\n"
            "  • They are dominated by the random starting point of training: the SAME task trained\n"
            "    twice from different starts gives UNRELATED add-ons (similarity 0.03), while\n"
            "    different tasks from the same start look alike (0.62).\n"
            "  • So you cannot reliably compare, average, or do arithmetic on add-ons across runs.\n"
            "  • A faint real task signal does exist once the shared 'starting-point' direction is\n"
            "    removed, and it lives more in the model's vision/language part ('what to grab') than\n"
            "    in the arm-motion part ('how to move') — which is intuitive.\n\n"
            "FOR A REAL SKILLS LIBRARY:\n"
            "  • Either start every skill from the SAME fixed point (so they're comparable), or\n"
            "  • compare/combine skills by what they DO (behavior), not by their raw numbers.\n\n"
            "THE NON-OBVIOUS TAKEAWAY:\n"
            "  A LoRA is a genuine functional skill, but NOT a faithful skill vector — and you only\n"
            "  find that out if you run the 'same task, different random start' control. Without it,\n"
            "  the raw numbers would have fooled us into 'the tasks are similar.'",
            subtitle="The useful, surprising part — and what it implies for building a skills library"),
        text_page("Honest limitations & glossary",
            "LIMITATIONS\n"
            "  • Error is measured offline (predicted vs demonstrated move), not as success on the\n"
            "    physical arm — a strong proxy, but a real robot trial is the gold standard.\n"
            "  • Only 4 objects, so no claims like 'all soft toys cluster.'\n"
            "  • The key 'same task, different start' control was run once (bottle); repeating it per\n"
            "    object would make it airtight. Bear/duck/hat held-out sets are small (5–20 eps).\n\n"
            "GLOSSARY\n"
            "  • AI model (π0.5): ~3.3B-number network mapping camera images + arm position → next move.\n"
            "  • Full fine-tune: retrain ALL the model's numbers (powerful, expensive).\n"
            "  • LoRA / skill add-on: a SMALL set of extra numbers added on a frozen model to\n"
            "    specialize it for one task — cheap, swappable.\n"
            "  • Error (degrees): how far the predicted joint angles are from the human demo (lower\n"
            "    is better; ~1° is very precise).\n"
            "  • Held-out: test data never seen in training (honest test of generalization).\n"
            "  • Similarity (0–1): how alike two sets of numbers point (1 identical, 0 unrelated).\n"
            "  • Random start / seed: the random initial values training begins from.\n"
            "  • Domain shift: changing lighting/color/blur to test robustness to a new environment.\n\n"
            "Full technical write-up: REPORT.md.  All models (base + add-ons) and the dataset are\n"
            "public on HuggingFace; code is in model/ and model/cluster/."),
    ]
    with PdfPages(a.pdf) as pdf:
        for f in pages:
            pdf.savefig(f); plt.close(f)
    print(f"wrote {a.pdf}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()
