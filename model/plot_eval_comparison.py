#!/usr/bin/env python3
"""ROOT (PyROOT) comparison figures: GSE vs full fine-tune offline eval.

Reads two ``eval_offline.py`` result JSONs and renders scientific comparison
plots with CERN ROOT — per-joint accuracy, headline error, inference latency,
the per-frame error distribution, and per-episode consistency. Writes one PNG
per figure, a 2×2 dashboard, and a combined multi-page PDF.

  micromamba run -n root python plot_eval_comparison.py \\
      eval-gse.json eval-full.json --outdir .
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import ROOT

JN = ["j1", "j2", "j3", "j4", "j5", "j6"]
DEG = 180.0 / np.pi
_keep = []  # hold ROOT objects so PyROOT's GC doesn't blank the canvases


def setup_style():
    ROOT.gROOT.SetBatch(True)
    s = ROOT.TStyle("farm", "farm")
    s.SetOptStat(0)
    s.SetOptTitle(1)
    s.SetCanvasColor(0)
    s.SetPadColor(0)
    s.SetFrameFillColor(0)
    s.SetPadGridX(True)
    s.SetPadGridY(True)
    s.SetGridColor(ROOT.kGray + 1)
    s.SetGridStyle(3)
    s.SetPadLeftMargin(0.13)
    s.SetPadBottomMargin(0.14)
    s.SetPadTopMargin(0.11)
    s.SetPadRightMargin(0.05)
    for ax in "XYZ":
        s.SetTitleFont(42, ax)
        s.SetLabelFont(42, ax)
        s.SetTitleSize(0.052, ax)
        s.SetLabelSize(0.045, ax)
    s.SetTitleFont(42, "")
    s.SetTitleSize(0.055, "")
    s.SetLegendFont(42)
    s.SetLegendTextSize(0.040)
    s.SetHistLineWidth(2)
    ROOT.gROOT.SetStyle("farm")
    ROOT.gROOT.ForceStyle()


def col(hexstr):
    return ROOT.TColor.GetColor(hexstr)


def grouped_bars(cats, gse, full, title, ytitle, fname, c_gse, c_full, fmt="{:.2f}", pdf=None):
    """Side-by-side bars per category with value labels + legend."""
    n = len(cats)
    hg = ROOT.TH1F("hg_" + os.path.basename(fname), title, n, 0, n)
    hf = ROOT.TH1F("hf_" + os.path.basename(fname), title, n, 0, n)
    for i, c in enumerate(cats):
        hg.SetBinContent(i + 1, gse[i])
        hf.SetBinContent(i + 1, full[i])
        hg.GetXaxis().SetBinLabel(i + 1, c)
    for h, cc in ((hg, c_gse), (hf, c_full)):
        h.SetFillColor(cc)
        h.SetLineColor(cc)
        h.SetBarWidth(0.40)
    hg.SetBarOffset(0.07)
    hf.SetBarOffset(0.53)
    ymax = max(max(gse), max(full)) * 1.28 or 1.0
    hg.SetMaximum(ymax)
    hg.SetMinimum(0.0)
    hg.GetYaxis().SetTitle(ytitle)
    hg.GetXaxis().SetLabelSize(0.055)
    cv = ROOT.TCanvas("c_" + os.path.basename(fname), title, 920, 620)
    hg.Draw("bar2")
    hf.Draw("bar2 same")
    lat = ROOT.TLatex()
    lat.SetTextSize(0.030)
    lat.SetTextAlign(21)
    for i in range(n):
        lat.SetTextColor(c_gse)
        lat.DrawLatex(i + 0.27, gse[i] + ymax * 0.015, fmt.format(gse[i]))
        lat.SetTextColor(c_full)
        lat.DrawLatex(i + 0.73, full[i] + ymax * 0.015, fmt.format(full[i]))
    leg = ROOT.TLegend(0.68, 0.79, 0.93, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(hg, "GSE", "f")
    leg.AddEntry(hf, "full FT", "f")
    leg.Draw()
    cv.RedrawAxis()
    cv.SaveAs(fname)
    if pdf:
        cv.Print(pdf)
    _keep.extend([hg, hf, cv, leg, lat])
    return cv


def dist_compare(gse_vals, full_vals, title, xtitle, fname, c_gse, c_full, pdf=None):
    """Overlaid normalized histograms of per-sample error."""
    hi = max(np.percentile(gse_vals, 99), np.percentile(full_vals, 99)) * 1.15
    hg = ROOT.TH1F("dg_" + os.path.basename(fname), title, 45, 0, hi)
    hf = ROOT.TH1F("df_" + os.path.basename(fname), title, 45, 0, hi)
    for v in gse_vals:
        hg.Fill(v)
    for v in full_vals:
        hf.Fill(v)
    hg.Scale(1.0 / hg.Integral())
    hf.Scale(1.0 / hf.Integral())
    hg.SetLineColor(c_gse)
    hg.SetLineWidth(3)
    hg.SetFillColorAlpha(c_gse, 0.22)
    hf.SetLineColor(c_full)
    hf.SetLineWidth(3)
    hf.SetFillColorAlpha(c_full, 0.22)
    hg.GetXaxis().SetTitle(xtitle)
    hg.GetYaxis().SetTitle("fraction of samples")
    hg.SetMaximum(max(hg.GetMaximum(), hf.GetMaximum()) * 1.30)
    cv = ROOT.TCanvas("cd_" + os.path.basename(fname), title, 920, 620)
    hg.Draw("hist")
    hf.Draw("hist same")
    leg = ROOT.TLegend(0.55, 0.76, 0.93, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(hg, "GSE  (mean {:.2f} deg)".format(float(np.mean(gse_vals))), "f")
    leg.AddEntry(hf, "full FT  (mean {:.2f} deg)".format(float(np.mean(full_vals))), "f")
    leg.Draw()
    cv.RedrawAxis()
    cv.SaveAs(fname)
    if pdf:
        cv.Print(pdf)
    _keep.extend([hg, hf, cv, leg])
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs=2, help="two eval_offline JSONs (any order)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    data = {}
    for p in args.results:
        d = json.load(open(p))
        data[d.get("model") or os.path.basename(p)] = d
    if "gse" not in data or "full" not in data:
        raise SystemExit(f"need a 'gse' and a 'full' result; got {list(data)}")
    gse, full = data["gse"], data["full"]
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda f: os.path.join(args.outdir, f)  # noqa: E731

    setup_style()
    c_gse, c_full = col("#2563eb"), col("#dc2626")
    cvs = []

    # 1) per-joint MAE (deg)
    cvs.append(grouped_bars(JN, [v * DEG for v in gse["per_joint_mae_rad"]],
               [v * DEG for v in full["per_joint_mae_rad"]],
               "Per-joint action error on training episodes",
               "MAE (deg)", out("cmp_per_joint_mae.png"), c_gse, c_full))

    # 2) headline joint MAE: immediate vs full chunk (deg)
    cvs.append(grouped_bars(["next-step", "10-step chunk"],
               [gse["overall_joint_mae_rad"] * DEG, gse["chunk_joint_mae_rad"] * DEG],
               [full["overall_joint_mae_rad"] * DEG, full["chunk_joint_mae_rad"] * DEG],
               "Joint action error (aggregate)", "MAE (deg)",
               out("cmp_headline_mae.png"), c_gse, c_full))

    # 3) gripper MAE
    cvs.append(grouped_bars(["gripper"], [gse["gripper_mae"]], [full["gripper_mae"]],
               "Gripper error (0=open .. ~0.3=closed)", "MAE",
               out("cmp_gripper_mae.png"), c_gse, c_full, fmt="{:.4f}"))

    # 4) inference latency (ms): median + p90
    cvs.append(grouped_bars(["median", "p90"],
               [gse["latency_ms"]["median_ms"], gse["latency_ms"]["p90_ms"]],
               [full["latency_ms"]["median_ms"], full["latency_ms"]["p90_ms"]],
               "Inference latency per action-chunk", "ms",
               out("cmp_latency.png"), c_gse, c_full, fmt="{:.0f}"))

    # 5) per-frame joint-error distribution (deg)
    ge = np.array(gse["samples"]["joint_err_rad"]).ravel() * DEG
    fe = np.array(full["samples"]["joint_err_rad"]).ravel() * DEG
    cvs.append(dist_compare(ge, fe, "Per-frame joint error distribution",
               "joint error (deg)", out("cmp_error_dist.png"), c_gse, c_full))

    # 6) per-episode joint MAE (deg) — consistency across episodes/tasks
    eps = [e["name"].replace("episode_", "")[:13] for e in gse["episodes"]]
    gmae = [e["joint_mae_rad"] * DEG for e in gse["episodes"]]
    fmap = {e["name"]: e["joint_mae_rad"] * DEG for e in full["episodes"]}
    fmae = [fmap.get(e["name"], 0.0) for e in gse["episodes"]]
    cvs.append(grouped_bars(eps, gmae, fmae, "Per-episode joint error", "MAE (deg)",
               out("cmp_per_episode.png"), c_gse, c_full))

    # combined multi-page PDF (open on the first canvas, close on the last)
    pdf = out("gse_vs_full.pdf")
    for i, cv in enumerate(cvs):
        cv.Print(pdf + ("(" if i == 0 else ")" if i == len(cvs) - 1 else ""))
    print("wrote:")
    for f in ("cmp_per_joint_mae", "cmp_headline_mae", "cmp_gripper_mae",
              "cmp_latency", "cmp_error_dist", "cmp_per_episode"):
        print("  " + out(f + ".png"))
    print("  " + pdf)

    # console summary of the differences
    print("\n=== GSE vs full (training-episode fit) ===")
    print(f"  overall joint MAE:  GSE {gse['overall_joint_mae_rad']*DEG:.2f}\xb0   "
          f"full {full['overall_joint_mae_rad']*DEG:.2f}\xb0")
    print(f"  10-step chunk MAE:  GSE {gse['chunk_joint_mae_rad']*DEG:.2f}\xb0   "
          f"full {full['chunk_joint_mae_rad']*DEG:.2f}\xb0")
    print(f"  gripper MAE:        GSE {gse['gripper_mae']:.4f}   full {full['gripper_mae']:.4f}")
    print(f"  latency (median):   GSE {gse['latency_ms']['median_ms']:.0f} ms   "
          f"full {full['latency_ms']['median_ms']:.0f} ms")


if __name__ == "__main__":
    main()
