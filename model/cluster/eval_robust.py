#!/usr/bin/env python3
"""Domain-shift ROBUSTNESS eval for a trained π0.5 policy — in-process, no robot.

The standard offline eval (``eval_offline.py``) replays the *original* recorded
JPEGs, so it only ever measures fit on the training distribution — which is
exactly why it scored sub-1° while the live policy was blind in a different room
(see model/FINDINGS.md + the env-mismatch notes). This eval instead applies
**test-time appearance perturbations** to each frame before feeding it to the
policy, as a *proxy* for the living-room→plywood-room domain shift, and reports
how much the predicted action degrades under each.

For every (episode, sampled frame) it runs the policy under each perturbation
CONDITION and compares the immediate next action to the recorded ground truth
(actions are absolute joint targets, so ``action[t] == state[t+1]``). The
perturbation for a given (condition, episode, frame) is deterministically seeded,
so **every model sees byte-identical perturbed inputs** → a fair comparison.

Photometric perturbations hit both cameras (lighting/colour shift everywhere);
occlusion hits the base camera only (a person/object enters that view). The
"domain_combo" condition stacks several to mimic a realistic room change.

Caveat (be honest in any writeup): synthetic perturbations are a PROXY for the
real domain shift, not the real thing. The only true validation is live
deployment / target-environment demos. Still, a model that holds up here is far
more likely to transfer than one that collapses.

  uv run python eval_robust.py \\
      --config pi05_farm_uf850_gse \\
      --checkpoint-dir ~/farm-train/openpi/checkpoints/.../2999 \\
      --episodes-dir ~/farm-train/eval_episodes \\
      --model gse_robust --out eval-robust-gse_robust.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
# openpi / openpi_client imported lazily inside main() so the pure-numpy/PIL
# perturbation helpers below can be imported + unit-tested without a GPU/openpi.

JN = ["j1", "j2", "j3", "j4", "j5", "j6"]
DEG = 180.0 / np.pi

# Order matters — this is also the plot order (clean first, hardest last).
CONDITIONS = [
    "clean", "bright", "dark", "low_contrast",
    "hue_shift", "desaturate", "blur", "occlude", "noise", "domain_combo",
]


# ─────────────────────────── perturbations ───────────────────────────
# All operate on a HxWx3 uint8 array and return the same. `rng` is a seeded
# numpy Generator so a given (condition, episode, frame) is reproducible across
# models. `is_base` lets base-only effects (occlusion, crop) skip the wrist cam.

def _pil(a):           return Image.fromarray(a)
def _np(im):           return np.asarray(im, dtype=np.uint8)


def _hue_rotate(a: np.ndarray, deg: float) -> np.ndarray:
    hsv = np.asarray(_pil(a).convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(deg / 360.0 * 255)) % 256
    return _np(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"))


def _crop_zoom(a: np.ndarray, frac: float, dx: float, dy: float) -> np.ndarray:
    """Center-ish crop to `frac` then resize back — simulates a camera that is
    closer / repositioned. dx,dy in [-1,1] shift the crop window."""
    h, w = a.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    top = int((h - ch) * (0.5 + 0.5 * dy))
    left = int((w - cw) * (0.5 + 0.5 * dx))
    crop = _pil(a).crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)
    return _np(crop)


def perturb(a: np.ndarray, cond: str, rng: np.random.Generator, is_base: bool) -> np.ndarray:
    if cond == "clean":
        return a
    if cond == "bright":
        return _np(ImageEnhance.Brightness(_pil(a)).enhance(1.6))
    if cond == "dark":
        return _np(ImageEnhance.Brightness(_pil(a)).enhance(0.5))
    if cond == "low_contrast":
        return _np(ImageEnhance.Contrast(_pil(a)).enhance(0.5))
    if cond == "hue_shift":
        return _hue_rotate(a, 45.0)
    if cond == "desaturate":
        return _np(ImageEnhance.Color(_pil(a)).enhance(0.15))
    if cond == "blur":
        return _np(_pil(a).filter(ImageFilter.GaussianBlur(radius=2.5)))
    if cond == "occlude":
        if not is_base:
            return a
        h, w = a.shape[:2]
        bw, bh = int(w * 0.45), int(h * 0.55)
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        out = a.copy()
        out[y0:y0 + bh, x0:x0 + bw] = 110  # neutral gray block (a person/object)
        return out
    if cond == "noise":
        n = rng.normal(0, 20, a.shape)
        return np.clip(a.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if cond == "domain_combo":
        # A realistic room change: dimmer + colour/background shift + slight
        # defocus, plus a reposition (base cam only).
        out = ImageEnhance.Brightness(_pil(a)).enhance(0.72)
        out = ImageEnhance.Color(out).enhance(0.6)
        out = _np(out.filter(ImageFilter.GaussianBlur(radius=1.4)))
        out = _hue_rotate(out, 28.0)
        if is_base:
            out = _crop_zoom(out, 0.82, float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.3, 0.3)))
        return out
    raise ValueError(f"unknown condition {cond!r}")


# ─────────────────────────── data loading ───────────────────────────

def state_of(frame: dict) -> np.ndarray:
    return np.array(list(frame["joints"]) + [float(frame.get("gripper_pos", 0.0))], dtype=np.float32)


def load_224(epdir: str, cam: str, i: int) -> np.ndarray:
    p = os.path.join(epdir, "cameras", cam, f"{i:06d}.jpg")
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(a, 224, 224))


def _seed(cond: str, ep: str, t: int) -> int:
    # Stable across processes (Python's hash() is PYTHONHASHSEED-salted, which
    # would give each model a DIFFERENT random occlusion/noise → unfair). md5 of
    # the key makes the perturbation reproducible, so every model sees identical
    # inputs for a given (condition, episode, frame).
    h = hashlib.md5(f"{cond}|{ep}|{t}".encode()).hexdigest()
    return int(h[:8], 16)


# ─────────────────────────── eval ───────────────────────────

def main() -> None:
    global image_tools
    from openpi.policies import policy_config
    from openpi.training import config as _config
    from openpi_client import image_tools

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--model", default="", help="label for the JSON (e.g. gse_robust)")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--samples-per-episode", type=int, default=8)
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="comma list; subset of " + ",".join(CONDITIONS))
    args = ap.parse_args()

    conds = [c for c in args.conditions.split(",") if c]
    cfg = _config.get_config(args.config)
    print(f">>> loading policy: config={args.config}\n    ckpt={args.checkpoint_dir}", flush=True)
    t0 = time.time()
    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    print(f"    ready in {time.time()-t0:.0f}s; conditions={conds}", flush=True)

    epdirs = sorted(glob.glob(os.path.join(args.episodes_dir, "episode_*")))[: args.n_episodes]
    if not epdirs:
        print(f"!!! no episode_* under {args.episodes_dir}")
        return

    # accumulators: per condition → list of per-frame mean joint errs (rad),
    # per-joint err arrays, gripper errs; and per (task,condition) mean joint err.
    acc = {c: {"j": [], "pj": [], "g": []} for c in conds}
    task_acc: dict[str, dict[str, list]] = {}
    tinf: list[float] = []

    for ep in epdirs:
        meta = json.load(open(os.path.join(ep, "meta.json")))
        task = meta["description"]
        name = os.path.basename(ep)
        frames = [json.loads(l) for l in open(os.path.join(ep, "frames.jsonl")) if l.strip()]
        states = [state_of(f) for f in frames]
        n = len(states)
        if n <= 2:
            continue
        idxs = np.linspace(0, n - 2, min(args.samples_per_episode, n - 2)).astype(int)
        task_acc.setdefault(task, {c: [] for c in conds})
        for t in idxs:
            base0 = load_224(ep, "base", t)
            wrist0 = load_224(ep, "wrist", t)
            gt = states[t + 1]
            for c in conds:
                rng = np.random.default_rng(_seed(c, name, int(t)))
                obs = {
                    "observation/image": perturb(base0, c, rng, is_base=True),
                    "observation/wrist_image": perturb(wrist0, c, rng, is_base=False),
                    "observation/state": states[t],
                    "prompt": task,
                }
                tic = time.perf_counter()
                pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
                tinf.append(time.perf_counter() - tic)
                je = np.abs(pred[0, :6] - gt[:6])
                acc[c]["j"].append(float(je.mean()))
                acc[c]["pj"].append(je)
                acc[c]["g"].append(abs(float(pred[0, 6]) - float(gt[6])))
                task_acc[task][c].append(float(je.mean()))
        print(f"  {name} ({task[:30]!r}): done {len(idxs)} frames × {len(conds)} conds", flush=True)

    # summarize
    out_conditions = {}
    print("\n=== ROBUSTNESS (immediate next-step joint MAE, deg) ===")
    clean_deg = float(np.mean(acc["clean"]["j"])) * DEG if acc.get("clean", {}).get("j") else float("nan")
    for c in conds:
        j = np.array(acc[c]["j"])
        pj = np.stack(acc[c]["pj"]) if acc[c]["pj"] else np.zeros((1, 6))
        g = np.array(acc[c]["g"])
        mae_rad = float(j.mean())
        out_conditions[c] = {
            "joint_mae_rad": mae_rad,
            "joint_mae_deg": mae_rad * DEG,
            "grip_mae": float(g.mean()),
            "per_joint_mae_rad": [float(pj[:, k].mean()) for k in range(6)],
            "samples_joint_err_rad": j.tolist(),
        }
        delta = mae_rad * DEG - clean_deg
        print(f"  {c:13s}: {mae_rad*DEG:6.2f}°   (Δ vs clean {delta:+5.2f}°)   grip {g.mean():.3f}")

    tarr = np.array(tinf) * 1000.0
    steady = tarr[1:] if len(tarr) > 1 else tarr
    lat = {"first_ms": float(tarr[0]), "median_ms": float(np.median(steady)),
           "mean_ms": float(steady.mean()), "p90_ms": float(np.percentile(steady, 90))}

    result = {
        "model": args.model, "config": args.config, "checkpoint": args.checkpoint_dir,
        "n_episodes": len(epdirs), "samples_per_episode": args.samples_per_episode,
        "conditions": out_conditions,
        "per_task": {tk: {c: (float(np.mean(v[c])) * DEG if v[c] else None) for c in conds}
                     for tk, v in task_acc.items()},
        "latency_ms": lat,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
