#!/usr/bin/env python3
"""LoRA 'skills vector' analysis — do similar manipulation tasks yield similar LoRAs?

Each per-task LoRA, trained IDENTICALLY off the same full-FT base (same rank, steps,
LR, seed; ONLY the LoRA adapters trainable — the FFT base incl. the vision tower is
frozen and shared; only the task data differs), is a low-rank weight delta ΔW = A·B
added on top of the frozen FFT-56k model. We treat each LoRA as a point in "skill
space" and ask: are task LoRAs near-collinear or near-orthogonal? Does the structure
match task semantics? Where does task identity live (VLM tower vs action expert)? Is
the vector a property of the TASK or the data/seed?

Controls (named by convention in the .npz set):
  * <task>30 (seed 42)            — the equal-size vector set {bottle,bear,duck,hat}.
  * bottle30s1 (seed 1)           — SAME task, DIFFERENT init → the cosine ceiling
                                    and the shared-init baseline (subtract to attribute
                                    cross-task similarity to task, not seed/base).
  * bottle100 (seed 42, more data)— SAME task, MORE data → is the vector the task or
                                    the data amount?

EXACTNESS — the Gram matrix G[i,j] = <ΔW_i, ΔW_j>_F is computed EXACTLY and cheaply
from the low-rank factors, never materialising the ~3.3B-dim ΔW:
    <A_i B_i, A_j B_j>_F = Σ_lead Σ_{r,s} (A_i^T A_j)[r,s] (B_i B_j^T)[r,s]
(A=[...,in,r], B=[...,r,out]; both [..,r,r] intermediates are tiny). ΔW scale = α/rank
= 1 for both towers (rank=alpha: 16 VLM, 32 action — verified), so ΔW = A·B exactly.

Everything downstream (cosine, MDS embedding, clustering, shared-skill alignment,
skill-arithmetic) is derived from G — exact, no projection, no OOM.

  python model/analyze_lora_vectors.py --indir analysis/fftLoRA_report/vectors \
      --outdir analysis/fftLoRA_report
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re

import numpy as np

# Task metadata for descriptive interpretation + pretty labels (NOT a powered test).
TASK_META = {
    "bottle": {"material": "rigid",  "size": "large",  "color": "#dc2626"},
    "bear":   {"material": "plush",  "size": "large",  "color": "#16a34a"},
    "duck":   {"material": "rubber", "size": "small",  "color": "#eab308"},
    "hat":    {"material": "fabric", "size": "medium", "color": "#2563eb"},
}


def base_task(label: str) -> str:
    for t in TASK_META:
        if label.startswith(t):
            return t
    return re.sub(r"\d.*$", "", label) or label


# ───────────────────────── load + pair adapters ─────────────────────────

def load_adapters(npz_path: str) -> dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return {k.replace("|", "/"): d[k] for k in d.files if k not in ("label", "keys")}


def pair_sites(flat: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """site_base -> (A, B). Pairs every '...lora_a' with its '...lora_b'."""
    sites = {}
    for k in flat:
        if k.endswith("lora_a"):
            base = k[: -len("lora_a")]
            bk = base + "lora_b"
            if bk in flat:
                A, B = np.asarray(flat[k], np.float64), np.asarray(flat[bk], np.float64)
                if A.shape[-1] != B.shape[-2]:
                    raise ValueError(f"site {base}: A{A.shape} B{B.shape} rank axes don't align "
                                     "(expected A=[...,in,r], B=[...,r,out])")
                sites[base.rstrip("_/")] = (A, B)
    return sites


def site_group(site: str) -> tuple[str, str]:
    """(tower, kind). openpi stacks both Gemma experts under PaliGemma/llm; the
    ACTION expert (expert index 1) carries a '_1' suffix on its einsum/mlp module
    name (q_einsum_1, kv_einsum_1, qkv_einsum_1, attn_vec_einsum_1, mlp_1); the VLM
    expert (index 0) has the bare name. So detect the action expert by '_1', NOT by
    a substring 'action'/'expert' (which appears in NO adapter key)."""
    s = site.lower()
    tower = "action" if re.search(r"(q_einsum|kv_einsum|qkv_einsum|attn_vec_einsum|mlp)_1(\b|/|_)", s) else "vlm"
    kind = "ffn" if any(t in s for t in ("gating", "linear", "mlp")) else "attn"
    return tower, kind


def site_ip(Ai, Bi, Aj, Bj) -> float:
    """Exact <A_i B_i, A_j B_j>_F over all leading (layer/scan) axes, no ΔW."""
    AtA = np.einsum("...mr,...ms->...rs", Ai, Aj, optimize=True)   # contract in-dim
    BBt = np.einsum("...rn,...sn->...rs", Bi, Bj, optimize=True)   # contract out-dim
    return float(np.einsum("...rs,...rs->", AtA, BBt, optimize=True))


# ───────────────────────── Gram over a site subset ─────────────────────────

def gram(tasks, dwsites, sites):
    """Exact n×n Gram G[i,j]=<ΔW_i,ΔW_j> over `sites` (streamed; tiny memory)."""
    n = len(tasks)
    G = np.zeros((n, n))
    for s in sites:
        for i in range(n):
            Ai, Bi = dwsites[tasks[i]][s]
            for j in range(i, n):
                Aj, Bj = dwsites[tasks[j]][s]
                v = site_ip(Ai, Bi, Aj, Bj)
                G[i, j] += v
                if j != i:
                    G[j, i] += v
    return G


def cosine_of(G):
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return G / np.outer(d, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="dir of <task>.npz adapter dumps")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.indir, "*.npz")))
    tasks, dwsites = [], {}
    for p in paths:
        label = os.path.splitext(os.path.basename(p))[0]
        sites = pair_sites(load_adapters(p))
        dwsites[label] = sites
        tasks.append(label)
        print(f"  {label}: {len(sites)} adapter sites")
    if len(tasks) < 2:
        raise SystemExit("need ≥2 LoRAs to compare")
    common = sorted(set.intersection(*[set(dwsites[t]) for t in tasks]))
    g_act = [s for s in common if site_group(s)[0] == "action"]
    g_vlm = [s for s in common if site_group(s)[0] == "vlm"]
    g_attn = [s for s in common if site_group(s)[1] == "attn"]
    g_ffn = [s for s in common if site_group(s)[1] == "ffn"]
    assert g_act and g_vlm, ("tower split is broken — found action sites: %d, vlm sites: %d "
                             "(check the '_1' action-expert suffix heuristic)" % (len(g_act), len(g_vlm)))
    print(f">>> {len(tasks)} LoRAs · {len(common)} common sites "
          f"(vlm {len(g_vlm)} / action {len(g_act)} · attn {len(g_attn)} / ffn {len(g_ffn)})")

    G = gram(tasks, dwsites, common)
    cos = cosine_of(G)
    cos_vlm = cosine_of(gram(tasks, dwsites, g_vlm))
    cos_act = cosine_of(gram(tasks, dwsites, g_act))
    cos_attn = cosine_of(gram(tasks, dwsites, g_attn))
    cos_ffn = cosine_of(gram(tasks, dwsites, g_ffn))

    # shared-skill direction (all from the cosine matrix — unit-vector geometry)
    n = len(tasks)
    mean_norm = float(np.sqrt(max(cos.sum() / n**2, 0.0)))
    shared = {tasks[i]: float(cos[i].sum() / (n * mean_norm + 1e-12)) for i in range(n)}

    # same-task (control) vs cross-task cosines
    bt = [base_task(t) for t in tasks]
    same_pairs, cross_pairs = [], []
    for i, j in itertools.combinations(range(n), 2):
        (same_pairs if bt[i] == bt[j] else cross_pairs).append((tasks[i], tasks[j], float(cos[i, j])))
    same_ceiling = float(np.mean([c for *_, c in same_pairs])) if same_pairs else None
    cross_mean = float(np.mean([c for *_, c in cross_pairs])) if cross_pairs else None

    # skill arithmetic on the equal-size seed-42 set (exact, from unnormalized G)
    prim = [t for t in tasks if re.fullmatch(r"(bottle|bear|duck|hat)30", t)]

    def gip(a, b):  # <ΔW_a, ΔW_b>
        return G[tasks.index(a), tasks.index(b)]

    def diff_cos(a, b, c, d):  # cos(ΔW_a-ΔW_b, ΔW_c-ΔW_d)
        num = gip(a, c) - gip(a, d) - gip(b, c) + gip(b, d)
        na = np.sqrt(max(gip(a, a) - 2 * gip(a, b) + gip(b, b), 1e-30))
        nc = np.sqrt(max(gip(c, c) - 2 * gip(c, d) + gip(d, d), 1e-30))
        return float(num / (na * nc))
    analogies = {}
    if set(["bottle30", "bear30", "duck30", "hat30"]).issubset(tasks):
        for (a, b, c, d) in [("bottle30", "bear30", "duck30", "hat30"),
                             ("bottle30", "duck30", "bear30", "hat30"),
                             ("bottle30", "hat30", "bear30", "duck30")]:
            analogies[f"({a}-{b}) · ({c}-{d})"] = diff_cos(a, b, c, d)

    metrics = {
        "tasks": tasks,
        "delta_w_fro_norm": {t: float(np.sqrt(G[i, i])) for i, t in enumerate(tasks)},
        "cosine_full": cos.tolist(),
        "cosine_vlm": cos_vlm.tolist(),
        "cosine_action": cos_act.tolist(),
        "cosine_attn": cos_attn.tolist(),
        "cosine_ffn": cos_ffn.tolist(),
        "shared_skill_alignment": shared,
        "same_task_pairs": same_pairs,
        "cross_task_pairs": cross_pairs,
        "same_task_cosine_ceiling": same_ceiling,
        "cross_task_cosine_mean": cross_mean,
        "skill_analogies": analogies,
        "n_sites": len(common), "n_vlm": len(g_vlm), "n_action": len(g_act),
        "primary_set": prim,
        "note": ("n=4 distinct objects → material/size 'grouping' has no statistical power; "
                 "we report pairwise cosines descriptively. Cross-task similarity is interpreted "
                 "RELATIVE to the same-task/different-seed ceiling."),
    }
    with open(os.path.join(args.outdir, "lora_vector_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(os.path.join(args.outdir, "lora_vector_arrays.npz"),
                        tasks=np.array(tasks, dtype=object), gram=G, cosine_full=cos,
                        cosine_vlm=cos_vlm, cosine_action=cos_act,
                        sites=np.array(common, dtype=object))
    print(f">>> wrote lora_vector_metrics.json + lora_vector_arrays.npz to {args.outdir}")

    print("\n=== full-ΔW cosine ===")
    print("            " + "".join(f"{t[:9]:>10s}" for t in tasks))
    for i, t in enumerate(tasks):
        print(f"  {t[:11]:11s}" + "".join(f"{cos[i,j]:10.3f}" for j in range(n)))
    if same_ceiling is not None:
        print(f"\nsame-task cosine ceiling (control): {same_ceiling:.3f}")
    if cross_mean is not None:
        print(f"cross-task cosine mean:             {cross_mean:.3f}")
    print("shared-skill alignment:", {t: round(v, 3) for t, v in shared.items()})
    if analogies:
        print("skill analogies (diff-vector cosine):", {k: round(v, 3) for k, v in analogies.items()})
    try:
        _figures(args.outdir, tasks, cos, cos_vlm, cos_act, G, common, dwsites, shared)
    except Exception as exc:
        print(f"  (figure step skipped: {exc})")


def _figures(outdir, tasks, cos, cos_vlm, cos_act, G, sites, dwsites, shared):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    cols = [TASK_META.get(base_task(t), {}).get("color", "#666") for t in tasks]
    n = len(tasks)

    def heat(ax, M, title):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n)); ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(tasks, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 0.6 else "#111")
        ax.set_title(title, fontsize=11, fontweight="bold"); return im

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, M, ti in zip(axes, (cos, cos_vlm, cos_act),
                         ("ΔW cosine — full model", "VLM tower only", "action expert only")):
        im = heat(ax, M, ti)
    fig.colorbar(im, ax=axes, fraction=0.025, label="cosine similarity")
    fig.savefig(os.path.join(outdir, "fig_lora_cosine.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # classical MDS from the unit-vector Gram (cosine) + dendrogram (exact)
    D = np.clip(1 - cos, 0, 2); np.fill_diagonal(D, 0)
    J = np.eye(n) - np.ones((n, n)) / n
    Bc = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(Bc)
    XY = V[:, -2:] * np.sqrt(np.clip(w[-2:], 0, None))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    a1.scatter(XY[:, 0], XY[:, 1], c=cols, s=300, edgecolor="k", zorder=3)
    for k, t in enumerate(tasks):
        a1.annotate(t, (XY[k, 0], XY[k, 1]), fontsize=9, ha="center", va="bottom")
    a1.set_title("Skill-space embedding (classical MDS on exact ΔW cosine)", fontweight="bold")
    a1.grid(alpha=0.3)
    Z = linkage(squareform(D, checks=False), method="average")
    dendrogram(Z, labels=tasks, ax=a2, color_threshold=0.7 * D.max())
    a2.set_title("Hierarchical clustering of task LoRAs", fontweight="bold")
    a2.tick_params(axis="x", rotation=40)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_lora_embedding.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # per-site ‖ΔW‖ profile (top task-varying sites) + shared-skill bars
    P = np.array([[float(np.sqrt(max(site_ip(*dwsites[t][s], *dwsites[t][s]), 0)))
                   for s in sites] for t in tasks])
    var = P.std(0); top = np.argsort(var)[::-1][:30]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [3, 1]})
    im = a1.imshow(P[:, top] / (P[:, top].max(0, keepdims=True) + 1e-9), aspect="auto", cmap="magma")
    a1.set_yticks(range(n)); a1.set_yticklabels(tasks, fontsize=8)
    a1.set_xticks(range(len(top)))
    a1.set_xticklabels([sites[i].split("/")[-1][:16] for i in top], rotation=80, ha="right", fontsize=6)
    a1.set_title("Where each task adapts — top-30 task-varying sites (col-norm ‖ΔW‖)", fontweight="bold")
    fig.colorbar(im, ax=a1, fraction=0.02)
    a2.barh(range(n), [shared[t] for t in tasks], color=cols)
    a2.set_yticks(range(n)); a2.set_yticklabels(tasks, fontsize=8)
    a2.set_xlabel("alignment to shared direction"); a2.set_title("Common-skill fraction", fontweight="bold")
    a2.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_lora_profile.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f">>> wrote 3 figures to {outdir}")


if __name__ == "__main__":
    main()
