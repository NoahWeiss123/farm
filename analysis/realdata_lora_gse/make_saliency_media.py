#!/usr/bin/env python3
"""Render the VLA saliency NPZs into clean overlays: full-episode MP4s + a grid
of frames + a full-task-prompt vs "bottle"-prompt comparison for the PDF.

Saliency = gradient of the policy's predicted action w.r.t. the input image
(SmoothGrad): which pixels most change what the policy does next. Two prompt
conditions: the full task string vs the single word "bottle".

NPZs in raw/ (or here): saliency_<ep>.npz (full task), saliency_<ep>_bottle.npz.
Each: base_rgb uint8 (T,224,224,3), saliency float32 (T,224,224), fps, episode_id, task.
Outputs: saliency_episode_<ep>[_bottle].mp4, figs/saliency_frames.png, figs/saliency_compare.png
"""
from __future__ import annotations
import os, subprocess, sys, tempfile, glob
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
UP = 2
TURBO = (plt.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def find(name):
    for d in (os.path.join(HERE, "raw"), HERE):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def load(path):
    z = np.load(path, allow_pickle=True)
    return (z["base_rgb"], z["saliency"].astype(np.float32),
            float(z["fps"]) if "fps" in z else 30.0,
            str(z["episode_id"]) if "episode_id" in z else "?",
            str(z["task"]) if "task" in z else "")


def _edge_mask(h, w, border=18, floor=0.08):
    """Taper the outermost pixels to suppress the SigLIP patch-conv boundary
    artifact (spurious high gradients at the image edge), keeping interior signal."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(np.float32)
    return floor + (1 - floor) * np.clip(d / border, 0, 1)


def _content_mask(base, ramp=18):
    """0 on the resize_with_pad LETTERBOX bars AND a tapered band ~`ramp` px into
    the scene next to them (where SmoothGrad inflates the gradient because the
    bars are constant-black / off-distribution under noise). Ramps to 1 in the
    true scene interior. Removes the bar + boundary artifact, keeps real content."""
    rows_ok = base.astype(np.float32).mean(axis=(0, 2, 3)) > 8.0
    cols_ok = base.astype(np.float32).mean(axis=(0, 1, 3)) > 8.0
    def taper(ok):
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return ok.astype(np.float32)
        lo, hi = idx.min(), idx.max()
        d = np.minimum(np.arange(len(ok)) - lo, hi - np.arange(len(ok))).astype(np.float32)
        return np.where(ok, np.clip(d / ramp, 0, 1), 0.0)
    return np.outer(taper(rows_ok), taper(cols_ok)).astype(np.float32)


def norm(sal, base, sigma=11, pct=99.5):
    """Per-episode global robust normalization + per-frame blur + edge-artifact
    suppression + letterbox masking -> [0,1]."""
    b = np.stack([cv2.GaussianBlur(s.astype(np.float32), (0, 0), sigmaX=sigma) for s in sal])
    b = np.clip(b, 0, None) * _edge_mask(sal.shape[1], sal.shape[2]) * _content_mask(base)[None]
    n = np.clip(b / (np.percentile(b, pct) + 1e-9), 0, 1) ** 0.7
    return n


def overlay(frame, s01, amax=0.72):
    h, w = frame.shape[:2]
    heat = TURBO[np.clip((s01 * 255).astype(np.int32), 0, 255)]
    a = (s01 * amax)[..., None]
    out = np.clip(frame.astype(np.float32) * (1 - a) + heat.astype(np.float32) * a, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w * UP, h * UP), interpolation=cv2.INTER_CUBIC) if UP > 1 else out


def make_mp4(base, n, fps, epid, tag, label):
    T = base.shape[0]
    mp4 = os.path.join(HERE, f"saliency_episode_{epid}{tag}.mp4")
    with tempfile.TemporaryDirectory() as td:
        for i in range(T):
            ov = cv2.cvtColor(overlay(base[i], n[i]), cv2.COLOR_RGB2BGR)
            cv2.putText(ov, f"{label}  ep{epid}  {i+1}/{T}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(td, f"f{i:04d}.png"), ov)
        r = subprocess.run(["ffmpeg", "-y", "-framerate", f"{fps:.0f}", "-i", os.path.join(td, "f%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4],
                           capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg failed:", r.stderr[-800:]); return None
    print(f"wrote {mp4} ({os.path.getsize(mp4)//1024} KB)")
    return mp4


def main():
    epid = "1"
    full = find(f"saliency_{epid}.npz")
    bott = find(f"saliency_{epid}_bottle.npz")
    if not full and not bott:
        print("no saliency NPZ found"); sys.exit(1)

    data = {}
    if full:
        base, sal, fps, eid, task = load(full); n = norm(sal, base)
        data["task"] = (base, n, fps, eid, task)
        make_mp4(base, n, fps, eid, "", "full-task prompt")
    if bott:
        base_b, sal_b, fps_b, eid_b, _ = load(bott); n_b = norm(sal_b, base_b)
        data["bottle"] = (base_b, n_b, fps_b, eid_b, "bottle")
        make_mp4(base_b, n_b, fps_b, eid_b, "_bottle", "prompt: 'bottle'")

    # frame grid for the primary (full-task) condition
    if "task" in data:
        base, n, fps, eid, task = data["task"]; T = base.shape[0]
        picks = np.linspace(T * 0.08, T * 0.92, 6).astype(int)
        fig, ax = plt.subplots(2, 6, figsize=(16, 5.6))
        for c, i in enumerate(picks):
            ax[0, c].imshow(base[i]); ax[0, c].set_title(f"frame {i+1}", fontsize=8); ax[0, c].axis("off")
            ax[1, c].imshow(overlay(base[i], n[i])); ax[1, c].axis("off")
        fig.text(0.012, 0.74, "real frame", rotation=90, va="center", fontsize=10, weight="bold")
        fig.text(0.012, 0.27, "+ saliency", rotation=90, va="center", fontsize=10, weight="bold")
        fig.suptitle(f"Where the GSE + LoRA policy looks — episode {eid}: {task[:58]}\n"
                     f"(turbo = pixels whose change most affects the predicted action; SmoothGrad)", fontsize=12)
        fig.tight_layout(rect=[0.02, 0, 1, 0.93])
        fig.savefig(os.path.join(FIG, "saliency_frames.png"), dpi=130); plt.close(fig)
        print("wrote figs/saliency_frames.png")

    # comparison: full-task prompt vs "bottle" prompt, same frames
    if "task" in data and "bottle" in data:
        base, n, _, eid, _ = data["task"]; _, n_b, _, _, _ = data["bottle"]
        T = base.shape[0]; picks = np.linspace(T * 0.12, T * 0.9, 5).astype(int)
        fig, ax = plt.subplots(3, 5, figsize=(15, 9))
        for c, i in enumerate(picks):
            ax[0, c].imshow(base[i]); ax[0, c].set_title(f"frame {i+1}", fontsize=9); ax[0, c].axis("off")
            ax[1, c].imshow(overlay(base[i], n[i])); ax[1, c].axis("off")
            ax[2, c].imshow(overlay(base[i], n_b[i])); ax[2, c].axis("off")
        for r, lab in [(0, "real frame"), (1, "full-task prompt"), (2, "prompt: \"bottle\"")]:
            fig.text(0.013, 0.83 - r * 0.31, lab, rotation=90, va="center", fontsize=11, weight="bold")
        fig.suptitle(f"What the VLA finds significant — full task string vs the single word \"bottle\"  (episode {eid})\n"
                     f"saliency = gradient of the predicted action w.r.t. the image", fontsize=13)
        fig.tight_layout(rect=[0.03, 0, 1, 0.94])
        fig.savefig(os.path.join(FIG, "saliency_compare.png"), dpi=130); plt.close(fig)
        print("wrote figs/saliency_compare.png")


if __name__ == "__main__":
    main()
