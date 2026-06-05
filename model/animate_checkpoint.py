#!/usr/bin/env python3
"""Animate the error-vs-steps checkpoint figure: the three curves draw in from
left to right (markers appear as each curve reaches a checkpoint), then hold.
Matches the static styling of fig_checkpoint() in make_clean_figures.py.

  python model/animate_checkpoint.py --dir analysis/fftLoRA_report
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

STYLE = {
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
    "figure.facecolor": "white", "savefig.facecolor": "white",
}
FPS, DRAW, HOLD, N = 30, 95, 35, 260


def smoothstep(t):
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/fftLoRA_report")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = a.dir
    out = a.out or os.path.join(d, "clean_7_checkpoint.mp4")

    sel = json.load(open(os.path.join(d, "fft_sweep", "fft_base_selection.json")))
    ps = sel["per_step"]; steps = sorted(int(s) for s in ps)
    g = lambda k: np.array([ps[str(s)].get(k) for s in steps], dtype=float)
    x = np.array(steps, dtype=float) / 1000.0
    series = [
        (g("train_deg"),       "#8b9097", "o", "train"),
        (g("held_clean_deg"),  "#0f6d75", "s", "held-out"),
        (g("held_robust_deg"), "#a4432f", "^", "held-out, shifted"),
    ]
    xd = np.linspace(x.min(), x.max(), N)
    dense = [(np.interp(xd, x, y), c, mk, lab, y) for (y, c, mk, lab) in series]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.7, 4.4))
        ax.set_axisbelow(True)
        xpad = (x.max() - x.min()) * 0.03
        ax.set_xlim(x.min() - xpad, x.max() + xpad)
        ax.set_ylim(0.8, 3.15)
        ax.set_xlabel(r"Training steps  ($\times10^{3}$)")
        ax.set_ylabel(r"Mean joint error  (degrees)")
        ax.set_title("Error vs. training steps")
        ax.minorticks_on(); ax.grid(which="minor", visible=False)

        lines, marks = [], []
        for (yd, c, mk, lab, yraw) in dense:
            ax.plot([], [], "-", marker=mk, color=c, lw=1.5, ms=5.0, mfc=c, mec=c,
                    mew=0.8, label=lab)                                    # legend proxy
            (ln,) = ax.plot([], [], "-", color=c, lw=1.5, zorder=3, label="_nolegend_")
            (mo,) = ax.plot([], [], linestyle="none", marker=mk, ms=5.0, mfc=c, mec=c,
                            mew=0.8, color=c, zorder=4, label="_nolegend_")
            lines.append(ln); marks.append(mo)

        leg = ax.legend(loc="upper right", frameon=True, framealpha=1.0,
                        edgecolor="#bcbcbc", borderpad=0.7, handlelength=2.0, labelspacing=0.55)
        leg.get_frame().set_linewidth(0.7)
        fig.tight_layout()

        def init():
            for ln in lines: ln.set_data([], [])
            for mo in marks: mo.set_data([], [])
            return lines + marks

        def animate(f):
            frac = smoothstep(min(f, DRAW) / DRAW)
            k = max(1, int(round(frac * N)))
            cutx = xd[k - 1] + 1e-9
            for (yd, c, mk, lab, yraw), ln, mo in zip(dense, lines, marks):
                ln.set_data(xd[:k], yd[:k])
                m = x <= cutx
                mo.set_data(x[m], yraw[m])
            return lines + marks

        anim = FuncAnimation(fig, animate, init_func=init, frames=DRAW + HOLD,
                             interval=1000 / FPS, blit=True)
        writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
        anim.save(out, writer=writer, dpi=200)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
