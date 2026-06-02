"""Append ``GSEMergeWeightLoader`` to openpi's ``training/weight_loaders.py``.

This is the weight loader that lets a **plain LoRA config initialize off a
TRAINED GSE checkpoint** (instead of base π0.5). It loads the GSE checkpoint and
MERGES the generalized/specialized attention adapters + the FFN LoRA adapters
back into the dense base weights, then hands the merged dense tree to
``_merge_params`` so the target LoRA param tree gets:
  * its dense base (w / gating_einsum / linear / action expert / img tower /
    norms / projections) FROM the merged GSE-multiobject model, and
  * fresh ``lora_a`` / ``lora_b`` adapters kept at their init (``.*lora.*``).

Why a merge (and why it's exact). In the ``*_gse`` variant every adapter adds
``(x·A)·B`` to ``x·W`` with **no scaling** (gse.Einsum sums gen+specialists with
no scale; lora.FeedForward._dot applies no scale, and alpha==rank==16 anyway).
For ``axes=(-2,-1)`` the two-step low-rank equals ``x·(A@B)`` on the last two
axes (lora's ``_make_lora_eqns`` construction), so the effective weight is
exactly ``W_eff = W + A@B`` for FFN and ``W_eff = w + gen_a@gen_b +
Σ_E spec_a[e]@spec_b[e]`` for attention. The specialized experts are stacked on
axis 1 (gemma prepends the scan layer axis, so E sits after it). Merge runs in
float32 (adapters are float32, base is bf16) and casts each ``W_eff`` back to its
slot dtype. attn_vec_einsum is included — its adapter started at zero-init but
trained from there, so it carries real signal at step 5999.

Idempotent. Run AFTER patch_openpi_gse.py (which creates the GSE modules +
GSESVDWeightLoader this file sits beside). Syntax-validated only; the real check
is that initial training loss starts sane (a broken merge → NaN/huge loss).
"""
from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

SENTINEL = "class GSEMergeWeightLoader"

INSERT = '''

@dataclasses.dataclass(frozen=True)
class GSEMergeWeightLoader(WeightLoader):
    """Init a (LoRA) param tree off a TRAINED GSE checkpoint by merging its
    generalized/specialized attention adapters + FFN LoRA into the dense base,
    then keeping the target's fresh lora_a/lora_b at init. Exact additive
    low-rank merge (no scaling); see patch_openpi_gse_merge.py."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        import jax.numpy as jnp

        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat = flax.traverse_util.flatten_dict(loaded, sep="/")

        def mm(a, b):  # batched matmul over all leading dims, float32
            return jnp.matmul(jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32))

        n_attn = n_ffn = 0
        # attention GSE einsums: W_eff = w + gen_a@gen_b + sum_E spec_a@spec_b
        for key in [k for k in list(flat) if k.endswith("/w") and (k[:-1] + "gse_gen_a") in flat]:
            pre = key[:-1]
            w = jnp.asarray(flat[key], jnp.float32)
            w_eff = w + mm(flat[pre + "gse_gen_a"], flat[pre + "gse_gen_b"]) \\
                      + jnp.sum(mm(flat[pre + "gse_spec_a"], flat[pre + "gse_spec_b"]), axis=1)
            if w_eff.shape != tuple(flat[key].shape):
                raise ValueError(f"GSE merge shape mismatch at {key}: {w_eff.shape} != {flat[key].shape}")
            if not bool(jnp.all(jnp.isfinite(w_eff))):
                raise ValueError(f"GSE merge produced non-finite weights at {key}")
            flat[key] = np.asarray(w_eff).astype(flat[key].dtype)
            for s in ("gse_gen_a", "gse_gen_b", "gse_spec_a", "gse_spec_b"):
                flat.pop(pre + s, None)
            n_attn += 1
        # FFN plain LoRA: W_eff = W + a@b  (scale 1)
        for wname, la, lb in (("gating_einsum", "gating_einsum_lora_a", "gating_einsum_lora_b"),
                              ("linear", "linear_lora_a", "linear_lora_b")):
            for key in [k for k in list(flat) if k.endswith("/" + wname) and (k[: -len(wname)] + la) in flat]:
                pre = key[: -len(wname)]
                w_eff = jnp.asarray(flat[key], jnp.float32) + mm(flat[pre + la], flat[pre + lb])
                if w_eff.shape != tuple(flat[key].shape):
                    raise ValueError(f"FFN merge shape mismatch at {key}")
                if not bool(jnp.all(jnp.isfinite(w_eff))):
                    raise ValueError(f"FFN merge produced non-finite weights at {key}")
                flat[key] = np.asarray(w_eff).astype(flat[key].dtype)
                flat.pop(pre + la, None); flat.pop(pre + lb, None)
                n_ffn += 1
        for k in [k for k in list(flat) if k.endswith("gse_router")]:
            flat.pop(k, None)
        residual = [k for k in flat if "gse_" in k or "_lora_" in k]
        print(f"[GSEMerge] merged {n_attn} attention + {n_ffn} FFN adapter sites into dense base; "
              f"residual adapter keys={len(residual)}", flush=True)
        if residual:
            raise ValueError(f"GSE merge left adapter keys unmerged: {residual[:6]}…")
        merged = flax.traverse_util.unflatten_dict(flat, sep="/")
        return _merge_params(merged, params, missing_regex=".*lora.*")
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--openpi-root", type=Path, default=Path.home() / "farm-train" / "openpi")
    args = ap.parse_args()
    wl = args.openpi_root / "src/openpi/training/weight_loaders.py"
    if not wl.is_file():
        raise SystemExit(f"FATAL: not found: {wl}")
    src = wl.read_text()
    if SENTINEL in src:
        print(f"[gse-merge] weight_loaders.py already has GSEMergeWeightLoader — no-op")
        return 0
    if "def _merge_params(" not in src or "class GSESVDWeightLoader" not in src:
        raise SystemExit("FATAL: weight_loaders.py missing _merge_params / GSESVDWeightLoader — run patch_openpi_gse.py first.")
    wl.write_text(src.rstrip() + "\n" + INSERT)
    try:
        py_compile.compile(str(wl), doraise=True)
    except py_compile.PyCompileError as exc:
        raise SystemExit(f"FATAL: weight_loaders.py failed to compile after patch:\n{exc}") from exc
    print(f"[gse-merge] appended GSEMergeWeightLoader to {wl} (compiles clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
