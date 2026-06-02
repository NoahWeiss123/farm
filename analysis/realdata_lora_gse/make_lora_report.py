#!/usr/bin/env python3
"""Figures + metrics + PDF for the bottle-LoRA experiment: does a LoRA initialized
off the GSE multiobject model generalize to held-out bottle episodes better than
one initialized off base pi0.5?

Clean probe (no camera/domain confound): the LoRAs trained on multiobject[0:100]
(bottle->box); we eval on the SAME dataset's held-out bottle episodes
multiobject[100:299] (same camera + task, unseen episodes). The separate
farm_uf850_bottle set is a DIFFERENT camera config and is excluded.

Inputs in raw/: eval-LORA-<tag>.json (+ -raw.npz for the full ones).
  full (npz): gse_indist (GSE-init FIT), gse_heldout, base_heldout, gsemodel_heldout_fit
  sweep (json only): {gse,base}_heldout_{2000,4000,6000,8000}
Outputs: figs/*.png, metrics.json, farm_lora_gse_report.pdf
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw"); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
DEG = 180.0 / np.pi
JN = ["j1", "j2", "j3", "j4", "j5", "j6"]; PANEL = JN + ["gripper"]
C_REAL = "#111111"; C_PRED = "#e8000b"; C_LOOK = "#1f77b4"
C_GSE = "#2077b4"; C_BASE = "#ff7f0e"; C_MODEL = "#2ca02c"


def load_npz(tag):
    p = os.path.join(RAW, f"eval-LORA-{tag}-raw.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = {"pred": z["sample_pred"].astype(np.float64), "gt": z["sample_gt"].astype(np.float64),
         "state": z["sample_state"].astype(np.float64), "task": np.array([str(t) for t in z["sample_task"]]),
         "ep": np.asarray(z["sample_ep"]).astype(int), "loc": np.asarray(z["sample_loc"]).astype(int),
         "H": int(z["horizon"]), "fps": float(z["fps"]), "rolls": []}
    for r in json.loads(str(z["roll_meta"])):
        i = r["idx"]
        d["rolls"].append({**r, "real": z[f"roll{i}_real"].astype(np.float64),
                           "pred": z[f"roll{i}_pred"].astype(np.float64), "loc": z[f"roll{i}_loc"]})
    return d


def load_json(tag):
    p = os.path.join(RAW, f"eval-LORA-{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def metrics(d):
    pred, gt, state, H = d["pred"], d["gt"], d["state"], d["H"]
    aerr = np.abs(pred[:, :, :6] - gt[:, :, :6])
    pf_end = aerr[:, H - 1, :].mean(axis=1) * DEG
    m = {"n": int(pred.shape[0]), "end_mae": float(aerr[:, H - 1].mean() * DEG),
         "first_mae": float(aerr[:, 0].mean() * DEG),
         "per_joint_end": (aerr[:, H - 1].mean(axis=0) * DEG).tolist(),
         "step_curve": (aerr.mean(axis=(0, 2)) * DEG).tolist(),
         "pf_end": pf_end}
    for t in (2, 5, 10):
        m[f"acc{t}"] = float((pf_end <= t).mean())
    g = np.abs(pred[:, H - 1, 6] - gt[:, H - 1, 6]); m["grip_acc"] = float((g <= 0.1).mean())
    dr = (gt[:, H - 1, :6] - state[:, :6]) * DEG; dp = (pred[:, H - 1, :6] - state[:, :6]) * DEG
    m["disp_r"] = float(np.corrcoef(dr.flatten(), dp.flatten())[0, 1]); m["dr"], m["dp"] = dr, dp
    num = (dr * dp).sum(1); den = np.linalg.norm(dr, axis=1) * np.linalg.norm(dp, axis=1); good = den > 1e-6
    m["dir_cos"] = float((num[good] / den[good]).mean())
    vr = np.diff(gt[:, :, :6], axis=1) * DEG; vp = np.diff(pred[:, :, :6], axis=1) * DEG
    m["vel_r"] = float(np.corrcoef(vr.flatten(), vp.flatten())[0, 1])
    return m


# ---------- figures ----------
def fig_headtohead(M, outpath):
    labels, keys = [], []
    for k, lab in [("gsemodel_heldout_fit", "base GSE*"), ("gse_heldout", "GSE + LoRA")]:
        if k in M:
            keys.append(k); labels.append(lab)
    cols = {"gse_heldout": C_GSE, "gsemodel_heldout_fit": C_MODEL}
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    panels = [("horizon-end MAE (deg)\nlower=better", lambda m: m["end_mae"], False),
              ("within 5° (%)\nhigher=better", lambda m: m["acc5"] * 100, True),
              ("displacement r\nhigher=better", lambda m: m["disp_r"], True),
              ("direction cosine\nhigher=better", lambda m: m["dir_cos"], True)]
    for ax_i, (title, fn, hib) in zip(ax, panels):
        vals = [fn(M[k]) for k in keys]; x = np.arange(len(keys))
        ax_i.bar(x, vals, color=[cols[k] for k in keys])
        for i, v in enumerate(vals): ax_i.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax_i.set_xticks(x); ax_i.set_xticklabels(labels, fontsize=8, rotation=12); ax_i.set_title(title, fontsize=10)
        ax_i.grid(alpha=.3, axis="y")
    fig.suptitle("Held-out bottle (multiobject[100:299], same camera) — base GSE vs GSE + LoRA"
                 "   (*base GSE trained on these episodes = fit ceiling, not held-out)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_fit_vs_heldout(mfit, mhel, outpath):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    cats = ["first step\n(33ms)", "horizon end\n(333ms)"]; x = np.arange(2)
    ax[0].bar(x - .2, [mfit["first_mae"], mfit["end_mae"]], .4, label="fit (trained eps)", color="#9ecae1")
    ax[0].bar(x + .2, [mhel["first_mae"], mhel["end_mae"]], .4, label="held-out eps", color=C_GSE)
    for i, v in enumerate([mfit["first_mae"], mfit["end_mae"]]): ax[0].text(i - .2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate([mhel["first_mae"], mhel["end_mae"]]): ax[0].text(i + .2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(cats); ax[0].set_ylabel("MAE (deg)")
    ax[0].set_title("GSE + LoRA: fit vs held-out"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    # displacement scatter: fit (left half) vs held-out — show held-out (the informative one)
    for axx, m, ttl in [(ax[1], mfit, f"fit  disp r={mfit['disp_r']:.2f}"), (ax[2], mhel, f"held-out  disp r={mhel['disp_r']:.2f}")]:
        dr, dp = m["dr"], m["dp"]; lim = max(np.abs(dr).max(), np.abs(dp).max(), 1) * 1.05
        axx.plot([-lim, lim], [-lim, lim], "--", color="#888", lw=1)
        axx.scatter(dr.flatten(), dp.flatten(), s=5, alpha=.25, color=C_GSE)
        axx.set_xlim(-lim, lim); axx.set_ylim(-lim, lim); axx.set_title(ttl, fontsize=10)
        axx.set_xlabel("real Δangle 333ms (deg)"); axx.set_ylabel("predicted Δangle"); axx.grid(alpha=.3)
    fig.suptitle("GSE + LoRA — does it track motion on UNSEEN bottle episodes?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_ckpt_sweep(sweep, outpath):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for model, col, lab in [("gse", C_GSE, "GSE + LoRA")]:
        steps, mae, acc = [], [], []
        for s in (2000, 4000, 6000, 8000, 9999):
            key = f"{model}_heldout" if s == 9999 else f"{model}_heldout_{s}"
            j = sweep.get(key)
            if j:
                steps.append(s); mae.append(j["end_of_horizon"]["overall_joint_mae_deg"])
                acc.append(j["end_of_horizon"]["accuracy_within_deg"]["5.0"] * 100)
        if steps:
            ax[0].plot(steps, mae, "o-", color=col, label=lab, lw=2)
            ax[1].plot(steps, acc, "o-", color=col, label=lab, lw=2)
            bi = int(np.argmin(mae)); ax[0].scatter([steps[bi]], [mae[bi]], s=160, facecolors="none", edgecolors=col, lw=2, zorder=5)
    ax[0].set_xlabel("checkpoint step"); ax[0].set_ylabel("held-out MAE (deg)"); ax[0].set_title("held-out error vs checkpoint (circle = best)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("checkpoint step"); ax[1].set_ylabel("held-out within 5° (%)"); ax[1].set_title("held-out accuracy vs checkpoint")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("Checkpoint selection on held-out bottle — later ≠ better (the over-fit signature)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_traj(d, label, outpath):
    if not d["rolls"]:
        return False
    roll = d["rolls"][0]; real, pr, loc, H = roll["real"], roll["pred"], roll["loc"], d["H"]; T = real.shape[0]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12)); axes = axes.flatten()
    seg = max(1, len(loc) // 9)
    for p in range(7):
        ax = axes[p]; sc = 1.0 if p == 6 else DEG
        ax.plot(np.arange(T), real[:, p] * sc, color=C_REAL, lw=2, label="real demo", zorder=3)
        ax.plot(loc + 1, pr[:, 0, p] * sc, color=C_PRED, lw=1.3, alpha=.85, label="predicted next-angle", zorder=4)
        for i in range(0, len(loc), seg):
            xs = loc[i] + 1 + np.arange(H)
            ax.plot(xs, pr[i, :, p] * sc, color=C_LOOK, lw=1, alpha=.6, marker="o", ms=2,
                    label="10-step lookahead" if i == 0 else None, zorder=2)
        ax.set_title(PANEL[p], fontsize=10); ax.set_ylabel("gripper" if p == 6 else "deg", fontsize=8)
        ax.grid(alpha=.25); ax.tick_params(labelsize=7)
    axes[7].axis("off"); h, l = axes[0].get_legend_handles_labels()
    axes[7].legend(h, l, loc="center", fontsize=11, frameon=False, title=f"ep{roll['episode']} (held-out)")
    fig.suptitle(f"{label}\npredicted next-angle vs real movement on a HELD-OUT bottle episode", fontsize=13, y=.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(outpath, dpi=130); plt.close(fig); return True


def fig_horizon(mhel, mfit, outpath):
    H = len(mhel["step_curve"]); ms = (np.arange(1, H + 1)) / 30 * 1000
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(ms, mfit["step_curve"], "o-", color="#9ecae1", label="fit")
    ax[0].plot(ms, mhel["step_curve"], "s-", color=C_GSE, label="held-out")
    ax[0].set_xlabel("lookahead (ms)"); ax[0].set_ylabel("MAE (deg)"); ax[0].set_title("GSE + LoRA: error vs horizon"); ax[0].legend(); ax[0].grid(alpha=.3)
    e = np.sort(mhel["pf_end"]); cdf = np.arange(1, len(e) + 1) / len(e) * 100
    ax[1].plot(e, cdf, color=C_GSE, lw=2)
    for t, c in [(2, "#2ca02c"), (5, "#ff7f0e"), (10, "#d62728")]:
        ax[1].axvline(t, color=c, ls="--", lw=1); ax[1].annotate(f"{mhel[f'acc{t}']*100:.0f}% ≤{t}°", (t, 8), fontsize=8, color=c, rotation=90, va="bottom")
    ax[1].set_xlabel("held-out per-frame MAE (deg)"); ax[1].set_ylabel("cumulative %"); ax[1].set_title("held-out accuracy CDF"); ax[1].grid(alpha=.3); ax[1].set_xlim(0, min(25, e.max() * 1.05))
    fig.suptitle("GSE + LoRA on held-out bottle", fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(outpath, dpi=130); plt.close(fig)


def per_episode_mae(d):
    """{episode_index: horizon-end joint MAE (deg)} from a loaded NPZ dict."""
    pred, gt, ep, H = d["pred"], d["gt"], d["ep"], d["H"]
    return {int(e): float(np.abs(pred[ep == e, H - 1, :6] - gt[ep == e, H - 1, :6]).mean() * DEG)
            for e in sorted(set(ep.tolist()))}


def fig_per_episode(fit, outpath):
    """fit = {tag: {ep: mae}} for fit15_gse / fit15_base / fit15_gsemodel."""
    order = [(t, l, c) for t, l, c in [("fit15_gsemodel", "base GSE", C_MODEL),
             ("fit15_gse", "GSE + LoRA", C_GSE)] if t in fit]
    prim = fit.get("fit15_gse") or fit[order[0][0]]
    eps = sorted(prim, key=lambda e: prim[e])                      # sort by GSE-init MAE
    x = np.arange(len(eps)); w = 0.8 / max(len(order), 1)
    pv = np.array([prim[e] for e in eps]); mu, sd = pv.mean(), pv.std(); thr = mu + 2 * sd
    fig, ax = plt.subplots(2, 1, figsize=(13, 9.5), gridspec_kw={"height_ratios": [2, 1]})
    for i, (tag, lab, col) in enumerate(order):
        vals = [fit[tag].get(e, np.nan) for e in eps]
        ax[0].bar(x + (i - (len(order) - 1) / 2) * w, vals, w, label=lab, color=col)
        ax[0].axhline(np.nanmean(list(fit[tag].values())), color=col, ls="--", lw=1, alpha=.7)
    for j, e in enumerate(eps):
        if prim[e] > thr:
            ax[0].annotate(f"ep{e}\n{prim[e]:.1f}°", (x[j], prim[e]), fontsize=7, ha="center", va="bottom", color="#b00", weight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"ep{e}" for e in eps], fontsize=7, rotation=45)
    ax[0].set_ylabel("per-episode horizon-end MAE (deg)")
    ax[0].set_title(f"Per-episode fit on 15 WITHIN-TRAINING episodes  (dashed = each model's average; "
                    f"red = GSE+LoRA outliers > μ+2σ = {thr:.2f}°)", fontsize=11)
    ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
    rng = np.random.default_rng(0)
    for i, (tag, lab, col) in enumerate(order):
        vals = list(fit[tag].values())
        ax[1].scatter(rng.normal(i, .05, len(vals)), vals, color=col, alpha=.7, s=28)
        ax[1].scatter([i], [np.mean(vals)], color="k", marker="_", s=700, zorder=5)
        ax[1].text(i, max(vals), f" avg {np.mean(vals):.2f}°\n max {max(vals):.2f}°", ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([o[1] for o in order])
    ax[1].set_ylabel("per-episode MAE (deg)"); ax[1].set_title("distribution across the 15 episodes (— = average)"); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("Within-training fit — reproducing 15 episodes the LoRA trained on (base GSE vs GSE + LoRA)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(outpath, dpi=130); plt.close(fig)
    return {t: {"avg": float(np.mean(list(fit[t].values()))), "min": float(min(fit[t].values())),
                "max": float(max(fit[t].values())), "outlier_eps": [e for e in eps if prim[e] > thr]} for t, _, _ in order}


def main():
    # ---- within-training per-episode (the 15-episode fit test) ----
    FIT = {}
    for t in ("fit15_gse", "fit15_base", "fit15_gsemodel"):
        d = load_npz(t)
        if d: FIT[t] = per_episode_mae(d)
    perep_stats = {}
    if FIT:
        perep_stats = fig_per_episode(FIT, os.path.join(FIG, "per_episode_fit.png"))
        print("per-episode fit:", {k: (round(v["avg"], 2), "outliers", v["outlier_eps"]) for k, v in perep_stats.items()})

    D = {t: load_npz(t) for t in ["fit15_gse", "gse_indist", "gse_heldout", "base_heldout", "gsemodel_heldout_fit"]}
    D = {k: v for k, v in D.items() if v}
    M = {k: metrics(v) for k, v in D.items()}
    sweep = {}
    for model in ("gse", "base"):
        for s in (2000, 4000, 6000, 8000):
            j = load_json(f"{model}_heldout_{s}")
            if j: sweep[f"{model}_heldout_{s}"] = j
        j = load_json(f"{model}_heldout")
        if j: sweep[f"{model}_heldout"] = j
    print("loaded npz:", list(M.keys()))
    for k, m in M.items():
        print(f"  {k}: endMAE={m['end_mae']:.2f}  w5={m['acc5']*100:.0f}%  dispR={m['disp_r']:.3f}  dirCos={m['dir_cos']:.3f}")

    figs = []
    # ---- LEAD: within-training per-episode (the user's 15-episode test) ----
    if os.path.exists(os.path.join(FIG, "per_episode_fit.png")):
        figs.append(("per_episode_fit.png", "Per-episode fit on 15 episodes the LoRA trained on: per-episode MAE, each model's average (dashed), and GSE+LoRA outliers (red). Lower = closer reproduction of the demo."))
    if "fit15_gse" in D and fig_traj(D["fit15_gse"], "GSE + LoRA — a WITHIN-TRAINING episode", os.path.join(FIG, "traj_fit15_gse.png")):
        figs.append(("traj_fit15_gse.png", "GSE + LoRA reproducing a bottle episode it TRAINED on: predicted next-angle (red) vs the real demo (black) — near-perfect tracking on training data."))
    if os.path.exists(os.path.join(FIG, "saliency_frames.png")):
        figs.append(("saliency_frames.png", "What the GSE + LoRA policy looks at: SmoothGrad saliency (gradient of the predicted action w.r.t. the input image) over real frames from a within-training episode — bright = pixels that most change the action. Full episode video: saliency_episode_<id>.mp4 in this folder."))
    if os.path.exists(os.path.join(FIG, "saliency_compare.png")):
        figs.append(("saliency_compare.png", "Prompt ablation: what the VLA finds significant when given the FULL task string vs. just the single word \"bottle\" — same frames, same image, only the language prompt differs. Videos: saliency_episode_<id>.mp4 and saliency_episode_<id>_bottle.mp4."))
    # ---- then: held-out generalization comparison ----
    if "gse_heldout" in M:
        fig_headtohead(M, os.path.join(FIG, "headtohead.png")); figs.append(("headtohead.png", "Held-out generalization (multiobject[100:299], unseen episodes): base GSE vs GSE + LoRA (base GSE = fit ceiling)."))
    if "gse_indist" in M and "gse_heldout" in M:
        fig_fit_vs_heldout(M["gse_indist"], M["gse_heldout"], os.path.join(FIG, "fit_vs_heldout.png")); figs.append(("fit_vs_heldout.png", "GSE + LoRA: near-perfect on trained episodes, degrades on unseen ones — displacement scatter shows how much motion-tracking survives."))
        fig_horizon(M["gse_heldout"], M["gse_indist"], os.path.join(FIG, "horizon.png")); figs.append(("horizon.png", "GSE + LoRA: error growth across the chunk + held-out accuracy CDF."))
    if sweep:
        fig_ckpt_sweep(sweep, os.path.join(FIG, "ckpt_sweep.png")); figs.append(("ckpt_sweep.png", "Checkpoint selection: held-out error/accuracy vs training step across training steps (the GSE + LoRA)."))
    if "gse_heldout" in D and fig_traj(D["gse_heldout"], "GSE + LoRA", os.path.join(FIG, "traj_gse_heldout.png")):
        figs.append(("traj_gse_heldout.png", "GSE + LoRA: commanded next-angle vs real movement on a held-out bottle episode."))

    json.dump({k: {kk: vv for kk, vv in m.items() if not isinstance(vv, np.ndarray)} for k, m in M.items()},
              open(os.path.join(HERE, "metrics.json"), "w"), indent=2, default=float)

    # ---- PDF ----
    with PdfPages(os.path.join(HERE, "farm_lora_gse_report.pdf")) as pdf:
        _cover(pdf, M, sweep, perep_stats)
        _narr(pdf, os.path.join(HERE, "FINDINGS.md"))
        for fn, cap in figs:
            _page(pdf, os.path.join(FIG, fn), cap)
    print("wrote farm_lora_gse_report.pdf  (+ metrics.json, figs/)")


def _cover(pdf, M, sweep, perep=None):
    import textwrap
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(.5, .955, "Bottle LoRA off the GSE Multiobject Model — Real-Episode Eval", ha="center", fontsize=15, weight="bold")
    fig.text(.5, .925, "Does adding a bottle LoRA to the GSE model help? — base GSE vs GSE + LoRA",
             ha="center", fontsize=9.5, style="italic", color="#444")
    # ---- within-training per-episode summary (the 15-episode test) ----
    if perep:
        pe = [["WITHIN-TRAIN (15 eps)", "avg MAE", "best ep", "worst ep"]]
        nm = {"fit15_gsemodel": "base GSE", "fit15_gse": "GSE + LoRA"}
        for t in ("fit15_gsemodel", "fit15_gse"):
            if t in perep:
                s = perep[t]; pe.append([nm[t], f"{s['avg']:.2f}°", f"{s['min']:.2f}°", f"{s['max']:.2f}°"])
        axp = fig.add_axes([.07, .70, .86, .15]); axp.axis("off")
        tp = axp.table(cellText=pe, loc="center", cellLoc="center"); tp.auto_set_font_size(False); tp.set_fontsize(9.5); tp.scale(1, 1.6)
        for c in range(4): tp[(0, c)].set_facecolor("#dff0df"); tp[(0, c)].set_text_props(weight="bold")
    fig.text(.07, .655, "HELD-OUT generalization (unseen episodes, same camera):", fontsize=9.5, weight="bold", color="#333")
    rows = [["held-out [100:299]", "base GSE*", "GSE + LoRA"]]
    def g(k, f, fmt="{:.2f}", s=1.0):
        return "—" if k not in M else fmt.format(f(M[k]) * s)
    rows += [
        ["horizon-end MAE (deg) ↓", g("gsemodel_heldout_fit", lambda m: m["end_mae"]), g("gse_heldout", lambda m: m["end_mae"])],
        ["first-step MAE (deg) ↓", g("gsemodel_heldout_fit", lambda m: m["first_mae"]), g("gse_heldout", lambda m: m["first_mae"])],
        ["within 5° (%) ↑", g("gsemodel_heldout_fit", lambda m: m["acc5"], "{:.0f}", 100), g("gse_heldout", lambda m: m["acc5"], "{:.0f}", 100)],
        ["displacement r ↑", g("gsemodel_heldout_fit", lambda m: m["disp_r"], "{:.3f}"), g("gse_heldout", lambda m: m["disp_r"], "{:.3f}")],
        ["direction cosine ↑", g("gsemodel_heldout_fit", lambda m: m["dir_cos"], "{:.3f}"), g("gse_heldout", lambda m: m["dir_cos"], "{:.3f}")],
    ]
    ax = fig.add_axes([.07, .34, .86, .29]); ax.axis("off")
    t = ax.table(cellText=rows, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.55)
    for c in range(3): t[(0, c)].set_facecolor("#dfe7f3"); t[(0, c)].set_text_props(weight="bold")
    note = ("base GSE = pi05_farm_multiobject_gse (the GSE multiobject model, step-5999).  GSE + LoRA = "
            "pi05_farm_bottle_lora_gse (a bottle LoRA trained on multiobject[0:100] on top of that GSE model, via "
            "GSEMergeWeightLoader).  Eval = open-loop teacher-forced single-shot prediction. WITHIN-TRAINING = 15 of the "
            "100 bottle episodes (30 frames each). HELD-OUT = unseen episodes multiobject[100:299], same camera + task "
            "(the farm_uf850_bottle set is a different camera config → excluded). *base GSE trained on ALL these episodes, "
            "so its rows are a FIT ceiling, not a held-out number.")
    fig.text(.07, .26, "\n".join(textwrap.wrap(note, 112)), fontsize=8.5, va="top", color="#333")
    pdf.savefig(fig); plt.close(fig)


def _narr(pdf, md):
    import textwrap
    if not os.path.exists(md): return
    lines = []
    for para in open(md).read().splitlines(): lines += textwrap.wrap(para, 100) or [""]
    for pi in range(0, len(lines), 46):
        fig = plt.figure(figsize=(11, 8.5)); fig.text(.07, .95, "Findings" + (" (cont.)" if pi else ""), fontsize=13, weight="bold")
        fig.text(.07, .91, "\n".join(lines[pi:pi + 46]), fontsize=8.5, va="top", family="monospace"); pdf.savefig(fig); plt.close(fig)


def _page(pdf, path, cap):
    img = mpimg.imread(path); h, w = img.shape[:2]; fw = 10.5; fh = min(7.6, fw / (w / h))
    fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_axes([(1 - fw / 11) / 2, .12, fw / 11, fh / 8.5]); ax.imshow(img); ax.axis("off")
    fig.text(.5, .06, cap, ha="center", fontsize=9, color="#333", wrap=True); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
