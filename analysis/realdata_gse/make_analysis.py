#!/usr/bin/env python3
"""In-depth analysis of the multiobject GSE policy's action predictions on REAL
episodes — in-distribution (its own training set) and, when available,
out-of-distribution (an earlier held-out dataset).

Consumes the raw NPZ dumped by model/cluster/eval_train_endhorizon.py (full
predicted chunks vs the real future at every sampled frame + dense whole-episode
rollouts) and produces derived metrics + figures. NO GPU, NO fabricated data —
every number traces to a real demonstrated frame and a real model inference.

Usage:
    python make_analysis.py                 # whatever raw/*.npz are present
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
DEG = 180.0 / np.pi
JN = ["j1", "j2", "j3", "j4", "j5", "j6"]
PANEL = JN + ["gripper"]
C_REAL = "#111111"; C_PRED = "#e8000b"; C_LOOK = "#1f77b4"; C_OOD = "#ff7f0e"; C_IND = "#2077b4"

# datasets to look for: (tag, human label, json, npz)
CANDIDATES = [
    ("indist", "In-distribution (multiobject training set)", "eval-indist.json", "eval-indist-raw.npz"),
    ("ood",    "Out-of-distribution (held-out farm_uf850_bottle)", "eval-ood.json", "eval-ood-raw.npz"),
]


def load(tag, jpath, npath):
    d = {"tag": tag}
    d["json"] = json.load(open(os.path.join(RAW, jpath))) if os.path.exists(os.path.join(RAW, jpath)) else {}
    z = np.load(os.path.join(RAW, npath), allow_pickle=True)
    d["pred"] = z["sample_pred"].astype(np.float64)      # (N,H,7) rad (+gripper)
    d["gt"]   = z["sample_gt"].astype(np.float64)        # (N,H,7)
    d["state"]= z["sample_state"].astype(np.float64)     # (N,7)
    d["task"] = np.array([str(t) for t in z["sample_task"]])
    d["ep"]   = z["sample_ep"]; d["loc"] = z["sample_loc"]
    d["H"]    = int(z["horizon"]); d["fps"] = float(z["fps"])
    d["roll_meta"] = json.loads(str(z["roll_meta"]))
    d["rolls"] = []
    for r in d["roll_meta"]:
        i = r["idx"]
        d["rolls"].append({**r,
            "real": z[f"roll{i}_real"].astype(np.float64),    # (T,7)
            "pred": z[f"roll{i}_pred"].astype(np.float64),    # (m,H,7)
            "loc":  z[f"roll{i}_loc"]})                        # (m,)
    return d


def metrics(d):
    pred, gt, state, H = d["pred"], d["gt"], d["state"], d["H"]
    fps = d["fps"]
    aerr = np.abs(pred[:, :, :6] - gt[:, :, :6])             # (N,H,6) rad
    perframe_end = aerr[:, H - 1, :].mean(axis=1) * DEG       # (N,) mean-joint deg @ horizon end
    m = {}
    m["n_samples"] = int(pred.shape[0])
    m["end_mae_deg"] = float(aerr[:, H - 1, :].mean() * DEG)
    m["first_mae_deg"] = float(aerr[:, 0, :].mean() * DEG)
    m["chunk_mae_deg"] = float(aerr.mean() * DEG)
    m["per_joint_end_deg"] = (aerr[:, H - 1, :].mean(axis=0) * DEG).tolist()
    m["per_joint_first_deg"] = (aerr[:, 0, :].mean(axis=0) * DEG).tolist()
    m["step_curve_deg"] = (aerr.mean(axis=(0, 2)) * DEG).tolist()
    m["step_perjoint_deg"] = (aerr.mean(axis=0) * DEG).tolist()   # (H,6)
    for t in (2, 5, 10):
        m[f"acc_within_{t}"] = float((perframe_end <= t).mean())
    g_end = np.abs(pred[:, H - 1, 6] - gt[:, H - 1, 6])
    m["gripper_end_mae"] = float(g_end.mean())
    m["gripper_acc_0.1"] = float((g_end <= 0.1).mean())
    m["perframe_end_deg"] = perframe_end                      # array, for CDF
    # ---- displacement fidelity over the 333ms horizon (real motion, not absolute pose) ----
    dr = (gt[:, H - 1, :6] - state[:, :6]) * DEG             # (N,6) real displacement deg
    dp = (pred[:, H - 1, :6] - state[:, :6]) * DEG           # predicted displacement deg
    m["disp_real"] = dr; m["disp_pred"] = dp
    # per-joint R^2 of predicted vs real displacement
    r2 = []
    for j in range(6):
        x, y = dr[:, j], dp[:, j]
        ss_res = np.sum((y - x) ** 2); ss_tot = np.sum((x - x.mean()) ** 2)
        r2.append(float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan"))
    m["disp_r2_perjoint"] = r2
    # overall displacement correlation (Pearson) on the flattened 6-D
    xf, yf = dr.flatten(), dp.flatten()
    m["disp_pearson"] = float(np.corrcoef(xf, yf)[0, 1])
    # ---- direction agreement over the horizon (does it move the right way) ----
    eps = 0.5  # deg: ignore joints that barely move
    mask = np.abs(dr) > eps
    sign_agree = []
    for j in range(6):
        mj = mask[:, j]
        sign_agree.append(float((np.sign(dp[mj, j]) == np.sign(dr[mj, j])).mean()) if mj.sum() else float("nan"))
    m["dir_sign_agree_perjoint"] = sign_agree
    # per-sample cosine of the 6-vector displacement
    num = (dr * dp).sum(axis=1)
    den = np.linalg.norm(dr, axis=1) * np.linalg.norm(dp, axis=1)
    good = den > 1e-6
    m["dir_cosine_mean"] = float((num[good] / den[good]).mean())
    m["dir_cosine_median"] = float(np.median(num[good] / den[good]))
    # ---- velocity (per-step increments) match ----
    vr = np.diff(gt[:, :, :6], axis=1) * DEG                  # (N,H-1,6)
    vp = np.diff(pred[:, :, :6], axis=1) * DEG
    m["vel_pearson"] = float(np.corrcoef(vr.flatten(), vp.flatten())[0, 1])
    m["vel_real"] = vr; m["vel_pred"] = vp
    # ---- per task ----
    pt = {}
    for tk in sorted(set(d["task"])):
        sel = d["task"] == tk
        pt[tk] = {"n": int(sel.sum()),
                  "end_mae_deg": float(aerr[sel, H - 1, :].mean() * DEG),
                  "acc_5": float((perframe_end[sel] <= 5).mean()),
                  "dir_cos": float((num[sel & good] / den[sel & good]).mean()) if (sel & good).sum() else float("nan")}
    m["per_task"] = pt
    # ---- rollout per-joint teacher-forced MAE (immediate next-angle vs real) ----
    for r in d["rolls"]:
        real, pr, loc = r["real"], r["pred"], r["loc"]
        # immediate next-angle prediction pred[i,0] is the command for frame loc[i]+1
        nxt_idx = np.clip(loc + 1, 0, real.shape[0] - 1)
        err = np.abs(pr[:, 0, :6] - real[nxt_idx, :6]).mean() * DEG
        r["nextangle_mae_deg"] = float(err)
    return m


# ----------------------------- figures -----------------------------
def fig_trajectory_overlay(d, label, roll, outpath):
    """Real joint trajectory + the model's predicted next-angle + lookahead segments."""
    real, pr, loc, H, fps = roll["real"], roll["pred"], roll["loc"], d["H"], d["fps"]
    T = real.shape[0]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12)); axes = axes.flatten()
    seg_every = max(1, len(loc) // 9)                         # ~9 lookahead fans across the episode
    for p in range(7):
        ax = axes[p]
        is_grip = (p == 6)
        scale = 1.0 if is_grip else DEG
        x_all = np.arange(T)
        ax.plot(x_all, real[:, p] * scale, color=C_REAL, lw=2.0, label="real demo", zorder=3)
        # continuous predicted next-angle (pred[:,0]) at frame loc+1
        ax.plot(loc + 1, pr[:, 0, p] * scale, color=C_PRED, lw=1.3, alpha=0.85,
                label="predicted next-angle", zorder=4)
        # a handful of full 10-step lookahead fans
        for i in range(0, len(loc), seg_every):
            xs = loc[i] + 1 + np.arange(H)
            ax.plot(xs, pr[i, :, p] * scale, color=C_LOOK, lw=1.0, alpha=0.6,
                    marker="o", ms=2.2, zorder=2,
                    label="10-step lookahead (0.33s)" if i == 0 else None)
        ax.set_title(PANEL[p] + ("" if is_grip else f"   (rollout MAE {np.abs(pr[:,0,p]-real[np.clip(loc+1,0,T-1),p]).mean()*DEG:.2f}°)"),
                     fontsize=10)
        ax.set_ylabel("gripper (0=open)" if is_grip else "angle (deg)", fontsize=8)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        if p >= 5: ax.set_xlabel("episode frame (30 Hz)", fontsize=8)
    axes[7].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[7].legend(h, l, loc="center", fontsize=11, frameon=False,
                   title="ep%d · %s" % (roll["episode"], roll["task"][:38]))
    fig.suptitle(f"{label}\nmodel's commanded next-angle vs the real demonstrated movement",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_displacement_scatter(d, m, label, outpath):
    dr, dp, r2 = m["disp_real"], m["disp_pred"], m["disp_r2_perjoint"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8)); axes = axes.flatten()
    for j in range(6):
        ax = axes[j]
        lim = max(np.abs(dr[:, j]).max(), np.abs(dp[:, j]).max(), 1.0) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="#888", lw=1, ls="--", zorder=1)
        ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
        ax.scatter(dr[:, j], dp[:, j], s=9, alpha=0.45, color=C_IND, zorder=2)
        ax.set_title(f"{JN[j]}   R²={r2[j]:.3f}", fontsize=10)
        ax.set_xlabel("real Δangle over 333ms (deg)", fontsize=8)
        ax.set_ylabel("predicted Δangle (deg)", fontsize=8)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle(f"{label}\npredicted vs real joint displacement over the 10-step horizon "
                 f"(Pearson r={m['disp_pearson']:.3f}) — points on the diagonal = the model "
                 f"commands the demonstrated motion", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_error_vs_horizon(d, m, label, outpath):
    H, fps = d["H"], d["fps"]
    steps = np.arange(1, H + 1); ms = steps / fps * 1000
    sp = np.array(m["step_perjoint_deg"])                    # (H,6)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j in range(6):
        ax[0].plot(ms, sp[:, j], marker="o", ms=3, lw=1, label=JN[j])
    ax[0].plot(ms, m["step_curve_deg"], color="k", lw=2.4, marker="s", ms=4, label="overall")
    ax[0].set_xlabel("lookahead (ms ahead of observation)"); ax[0].set_ylabel("MAE (deg)")
    ax[0].set_title("error growth across the predicted chunk"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, ncol=2)
    # CDF of per-frame horizon-end error
    e = np.sort(m["perframe_end_deg"]); cdf = np.arange(1, len(e) + 1) / len(e)
    ax[1].plot(e, cdf * 100, color=C_IND, lw=2)
    for t, c in [(2, "#2ca02c"), (5, "#ff7f0e"), (10, "#d62728")]:
        ax[1].axvline(t, color=c, ls="--", lw=1)
        ax[1].annotate(f"{m[f'acc_within_{t}']*100:.0f}% ≤{t}°", (t, 8), fontsize=8, color=c, rotation=90, va="bottom")
    ax[1].set_xlabel("horizon-end per-frame MAE (deg)"); ax[1].set_ylabel("cumulative % of frames")
    ax[1].set_title("accuracy CDF (333ms-ahead prediction)"); ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, min(20, e.max() * 1.05))
    fig.suptitle(label, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_perjoint_and_direction(d, m, label, outpath):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(6)
    ax[0].bar(x - 0.2, m["per_joint_first_deg"], 0.4, label="first step (33ms)", color="#9ecae1")
    ax[0].bar(x + 0.2, m["per_joint_end_deg"], 0.4, label="horizon end (333ms)", color=C_IND)
    ax[0].set_xticks(x); ax[0].set_xticklabels(JN); ax[0].set_ylabel("MAE (deg)")
    ax[0].set_title("per-joint prediction error"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(x, [s * 100 for s in m["dir_sign_agree_perjoint"]], color="#74c476")
    ax[1].axhline(50, color="#888", ls="--", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(JN); ax[1].set_ylabel("% correct direction")
    ax[1].set_title(f"motion-direction agreement\n(mean cosine of Δ-vector = {m['dir_cosine_mean']:.3f})")
    ax[1].set_ylim(0, 100); ax[1].grid(alpha=0.3, axis="y")
    vr, vp = m["vel_real"].flatten(), m["vel_pred"].flatten()
    lim = np.percentile(np.abs(np.concatenate([vr, vp])), 99.5)
    ax[2].plot([-lim, lim], [-lim, lim], "--", color="#888", lw=1)
    ax[2].scatter(vr, vp, s=3, alpha=0.15, color="#6a51a3")
    ax[2].set_xlim(-lim, lim); ax[2].set_ylim(-lim, lim)
    ax[2].set_xlabel("real per-step velocity (deg/step)"); ax[2].set_ylabel("predicted velocity")
    ax[2].set_title(f"velocity match (Pearson r={m['vel_pearson']:.3f})"); ax[2].grid(alpha=0.3)
    fig.suptitle(label, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_per_task(d, m, label, outpath):
    pt = m["per_task"]; tasks = list(pt.keys())
    short = [t.replace("Picking up the ", "").replace("Pick up the ", "")[:26] for t in tasks]
    mae = [pt[t]["end_mae_deg"] for t in tasks]; acc = [pt[t]["acc_5"] * 100 for t in tasks]
    ns = [pt[t]["n"] for t in tasks]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); y = np.arange(len(tasks))
    ax[0].barh(y, mae, color=C_IND); ax[0].set_yticks(y); ax[0].set_yticklabels(short, fontsize=8)
    for i, (v, n) in enumerate(zip(mae, ns)): ax[0].text(v, i, f" {v:.2f}° (n={n})", va="center", fontsize=8)
    ax[0].set_xlabel("horizon-end MAE (deg)"); ax[0].set_title("per-task error"); ax[0].grid(alpha=0.3, axis="x")
    ax[1].barh(y, acc, color="#74c476"); ax[1].set_yticks(y); ax[1].set_yticklabels([])
    for i, v in enumerate(acc): ax[1].text(min(v, 92), i, f" {v:.1f}%", va="center", fontsize=8)
    ax[1].set_xlabel("% frames within 5°"); ax[1].set_title("per-task accuracy"); ax[1].set_xlim(0, 105); ax[1].grid(alpha=0.3, axis="x")
    fig.suptitle(label, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=130); plt.close(fig)


def fig_compare(dind, mind, dood, mood, outpath):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    cats = ["first step\n(33ms)", "horizon end\n(333ms)"]
    ind = [mind["first_mae_deg"], mind["end_mae_deg"]]
    ood = [mood["first_mae_deg"], mood["end_mae_deg"]]
    x = np.arange(2)
    ax[0].bar(x - 0.2, ind, 0.4, label="in-dist", color=C_IND)
    ax[0].bar(x + 0.2, ood, 0.4, label="OOD (held-out)", color=C_OOD)
    for i, v in enumerate(ind): ax[0].text(i - 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(ood): ax[0].text(i + 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(cats); ax[0].set_ylabel("MAE (deg)")
    ax[0].set_title("prediction error: fit vs generalization"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, axis="y")
    # accuracy bars
    accs = [("≤2°", "acc_within_2"), ("≤5°", "acc_within_5"), ("≤10°", "acc_within_10")]
    xa = np.arange(len(accs))
    ax[1].bar(xa - 0.2, [mind[k] * 100 for _, k in accs], 0.4, label="in-dist", color=C_IND)
    ax[1].bar(xa + 0.2, [mood[k] * 100 for _, k in accs], 0.4, label="OOD", color=C_OOD)
    ax[1].set_xticks(xa); ax[1].set_xticklabels([a for a, _ in accs]); ax[1].set_ylabel("% of frames")
    ax[1].set_title("horizon-end accuracy"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, axis="y"); ax[1].set_ylim(0, 105)
    # error vs horizon both
    H = dind["H"]; ms = np.arange(1, H + 1) / dind["fps"] * 1000
    ax[2].plot(ms, mind["step_curve_deg"], marker="o", color=C_IND, lw=2, label="in-dist")
    ax[2].plot(ms, mood["step_curve_deg"], marker="s", color=C_OOD, lw=2, label="OOD")
    ax[2].set_xlabel("lookahead (ms)"); ax[2].set_ylabel("MAE (deg)")
    ax[2].set_title("error growth: fit vs generalization"); ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)
    fig.suptitle("Multiobject GSE — in-distribution fit vs out-of-distribution generalization",
                 fontsize=13); fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=130); plt.close(fig)


def task_short(t):
    return t.replace("Picking up the ", "").replace("Pick up the ", "")


def main():
    present = [(tag, lab, j, n) for tag, lab, j, n in CANDIDATES if os.path.exists(os.path.join(RAW, n))]
    if not present:
        print("no raw NPZ found in", RAW); sys.exit(1)
    data, mets = {}, {}
    for tag, lab, j, n in present:
        print(f">>> loading {tag}: {n}")
        d = load(tag, j, n); m = metrics(d)
        data[tag] = (d, lab); mets[tag] = m
        print(f"    n={m['n_samples']}  end-MAE={m['end_mae_deg']:.2f}°  first-MAE={m['first_mae_deg']:.2f}°  "
              f"within5°={m['acc_within_5']*100:.1f}%  dispR(pearson)={m['disp_pearson']:.3f}  "
              f"dir-cos={m['dir_cosine_mean']:.3f}  vel-r={m['vel_pearson']:.3f}")

    # per-dataset figures
    for tag, (d, lab) in data.items():
        m = mets[tag]
        # one trajectory overlay per task (first rollout of each task)
        seen = set()
        for r in d["rolls"]:
            if r["task"] in seen: continue
            seen.add(r["task"])
            fig_trajectory_overlay(d, lab, r, os.path.join(FIG, f"traj_{tag}_{task_short(r['task'])[:16].strip().replace(' ','_')}.png"))
        fig_displacement_scatter(d, m, lab, os.path.join(FIG, f"disp_{tag}.png"))
        fig_error_vs_horizon(d, m, lab, os.path.join(FIG, f"horizon_{tag}.png"))
        fig_perjoint_and_direction(d, m, lab, os.path.join(FIG, f"perjoint_{tag}.png"))
        fig_per_task(d, m, lab, os.path.join(FIG, f"pertask_{tag}.png"))
        print(f"    figures for {tag} written")

    if "indist" in data and "ood" in data:
        fig_compare(data["indist"][0], mets["indist"], data["ood"][0], mets["ood"],
                    os.path.join(FIG, "compare_indist_vs_ood.png"))
        print("    comparison figure written")

    # dump a compact metrics.json (arrays stripped)
    def clean(m):
        return {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
    json.dump({t: clean(mm) for t, mm in mets.items()},
              open(os.path.join(HERE, "metrics.json"), "w"), indent=2, default=float)
    print(">>> wrote metrics.json")
    return data, mets


if __name__ == "__main__":
    main()
