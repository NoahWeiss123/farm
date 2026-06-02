#!/usr/bin/env python3
"""Extract just the LoRA adapter tensors (lora_a / lora_b) from a trained openpi
checkpoint into a compact .npz, so the heavy 'skills vector' analysis can run
locally with no GPU and no 12GB base-weight load.

A gemma_2b_lora / gemma_300m_lora checkpoint stores the frozen dense base PLUS,
at every adapted site, a low-rank pair whose names end in ``..._lora_a`` and
``..._lora_b`` (FFN: gating_einsum/linear; attention: q/k/v/o einsums; in both the
PaliGemma VLM and the action expert). The effective weight delta at a site is
ΔW = a @ b (matmul on the last two axes; leading layer/expert axes are batched) —
the exact additive low-rank form openpi uses (see patch_openpi_gse_merge.py). We
save the raw a/b here and compute ΔW downstream.

Run in the openpi container (needs openpi for restore_params):

  uv run python extract_lora_adapters.py \
      --checkpoint-dir .../checkpoints/pi05_fftlora/fftlora_bottle35/12000 \
      --label bottle35 --out ~/farm-train/lora_vectors/bottle35.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, help="dir containing params/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import flax.traverse_util as ftu
    from openpi.models import model as _model
    from openpi.shared import download

    params_path = args.checkpoint_dir.rstrip("/")
    if not params_path.endswith("params"):
        params_path = os.path.join(params_path, "params")
    print(f">>> restoring params from {params_path}", flush=True)
    params = _model.restore_params(download.maybe_download(params_path), restore_type=np.ndarray)
    flat = ftu.flatten_dict(params, sep="/")

    lora = {k: np.asarray(v) for k, v in flat.items() if k.endswith("lora_a") or k.endswith("lora_b")}
    if not lora:
        # some openpi versions name them ".../lora/a" etc. — fall back to substring.
        lora = {k: np.asarray(v) for k, v in flat.items() if "lora" in k.lower() and v.ndim >= 2}
    if not lora:
        raise SystemExit("no LoRA adapter keys found — is this a *_lora checkpoint?")

    # Report the adapter inventory: site → (a shape, b shape).
    sites = {}
    for k in lora:
        if k.endswith("_lora_a") or k.endswith("lora_a"):
            base = k[: -len("lora_a")].rstrip("_/")
            sites.setdefault(base, {})["a"] = lora[k].shape
        elif k.endswith("_lora_b") or k.endswith("lora_b"):
            base = k[: -len("lora_b")].rstrip("_/")
            sites.setdefault(base, {})["b"] = lora[k].shape
    n_params = int(sum(v.size for v in lora.values()))
    print(f">>> {len(lora)} adapter tensors at {len(sites)} sites, {n_params/1e6:.2f}M adapter params")
    for s in sorted(sites)[:8]:
        print(f"    {s}: a={sites[s].get('a')} b={sites[s].get('b')}")
    if len(sites) > 8:
        print(f"    … (+{len(sites)-8} more sites)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Save flattened (key→array) plus the label; keys keep '/' so downstream pairs a/b.
    np.savez_compressed(args.out, label=np.array(args.label, dtype=object),
                        keys=np.array(list(lora.keys()), dtype=object),
                        **{k.replace("/", "|"): v for k, v in lora.items()})
    print(f">>> wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
