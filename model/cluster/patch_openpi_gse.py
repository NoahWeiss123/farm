"""Install GSE (Generalized & Specialized Experts) into the openpi checkout.

Idempotent, self-verifying (py_compiles every file it touches). Run this BEFORE
``patch_openpi_config_pi05_gse.py`` — ``setup.sh`` does both. Does four things:

1. Copies ``openpi_gse.py`` (staged beside this script) → ``models/gse.py``.
2. Patches ``models/gemma.py``: imports gse, registers the ``gemma_2b_gse`` /
   ``gemma_300m_gse`` variants, and routes the existing ``lora.Einsum`` /
   ``lora.FeedForward`` call sites through a dispatcher that picks the GSE
   module when the config is a ``gse.GSEConfig`` (attention) and plain LoRA
   otherwise (FFN). No call site is hand-edited — a single ``replace`` swaps the
   constructor name, and recursion-safe aliases keep the LoRA path intact.
3. Patches ``models/pi0_config.py`` ``get_freeze_filter`` to treat "gse" like
   "lora": freeze the VLM backbone, keep the GSE adapters + action expert
   trainable.
4. Appends ``GSESVDWeightLoader`` to ``training/weight_loaders.py`` — it
   SVD-initializes the attention-input GSE adapters and adjusts the backbone.

See ``openpi_gse.py`` for the method. NOTE: syntax-validated only (no GPU here);
run the smoke test in ``model/FINDINGS.md`` before a full training run.
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

GEMMA_IMPORT_ANCHOR = "import openpi.models.lora as lora\n"
GEMMA_DISPATCHER = '''import openpi.models.lora as lora
import openpi.models.gse as gse

# GSE dispatch: route the Gemma einsum / ffn sites through GSE when their config
# is a GSEConfig (attention in the *_gse variants), else through plain LoRA.
# Aliases (no trailing "(") are recursion-safe against the replace below.
_LoraEinsum = lora.Einsum
_LoraFeedForward = lora.FeedForward


def _gse_einsum(*, lora_config=None, **kw):
    if isinstance(lora_config, gse.GSEConfig):
        return gse.Einsum(gse_config=lora_config, **kw)
    return _LoraEinsum(lora_config=lora_config, **kw)


def _gse_feedforward(*, lora_config=None, **kw):
    if isinstance(lora_config, gse.GSEConfig):
        return gse.FeedForward(gse_config=lora_config, **kw)
    return _LoraFeedForward(lora_config=lora_config, **kw)
'''

GEMMA_VARIANT_OLD = (
    'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]'
)
GEMMA_VARIANT_NEW = (
    'Variant = Literal[\n'
    '    "dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora",\n'
    '    "gemma_2b_gse", "gemma_300m_gse",\n'
    ']'
)

GEMMA_GET_CONFIG_ANCHOR = '    raise ValueError(f"Unknown variant: {variant}")'
GEMMA_GET_CONFIG_INSERT = '''    if variant == "gemma_2b_gse":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": gse.GSEConfig(generalized_rank=2, num_specialized=7, expert_rank=2),
                "ffn": lora.LoRAConfig(rank=16, alpha=16.0),
            },
        )
    if variant == "gemma_300m_gse":
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": gse.GSEConfig(generalized_rank=2, num_specialized=7, expert_rank=2),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),
            },
        )
'''

# pi0_config.get_freeze_filter — treat "gse" like "lora".
FREEZE_EDITS = [
    (
        'if "lora" in self.paligemma_variant:',
        'if "lora" in self.paligemma_variant or "gse" in self.paligemma_variant:',
    ),
    (
        "if \"lora\" not in self.action_expert_variant:",
        'if "lora" not in self.action_expert_variant and "gse" not in self.action_expert_variant:',
    ),
    (
        'elif "lora" in self.action_expert_variant:',
        'elif "lora" in self.action_expert_variant or "gse" in self.action_expert_variant:',
    ),
    (
        'nnx.Not(nnx_utils.PathRegex(".*lora.*")),',
        'nnx.Not(nnx_utils.PathRegex(".*(lora|gse).*")),',
    ),
]

WEIGHT_LOADER_SENTINEL = "class GSESVDWeightLoader"
WEIGHT_LOADER_INSERT = '''

@dataclasses.dataclass(frozen=True)
class GSESVDWeightLoader(WeightLoader):
    """GSE init: load a base checkpoint, then SVD-initialize the attention-input
    GSE adapters (generalized = leading singular components, specialized =
    residual blocks) and subtract them from the frozen backbone (paper Eq. 12).

    Only the qkv / q / kv projections are SVD-initialized — the two-step
    low-rank reconstructs a weight exactly only when no leading axis is
    contracted, which holds there but not for the attention-output projection
    (``attn_vec_einsum``), whose adapter is left at its zero init (a plain
    trainable-from-zero delta). FFN uses plain LoRA (also zero init).
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        import jax.numpy as jnp

        from openpi.models import gse as _gse

        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        merged = _merge_params(loaded_params, params, missing_regex=".*(lora|gse).*")
        flat = flax.traverse_util.flatten_dict(merged, sep="/")
        for key in list(flat):
            if not key.endswith("/w"):
                continue
            prefix = key[:-1]  # ".../<einsum>/"
            gen_a_key = prefix + "gse_gen_a"
            if gen_a_key not in flat:
                continue  # not a GSE einsum
            if "attn_vec_einsum" in key:
                continue  # output projection: keep zero init (no valid SVD recon)
            rg = int(flat[gen_a_key].shape[-1])
            spec_a = flat[prefix + "gse_spec_a"]
            e, d = int(spec_a.shape[0]), int(spec_a.shape[-1])
            ga, gb, sa, sb, w_adj = _gse.svd_init_factors(jnp.asarray(flat[key]), (-2, -1), rg, e, d)
            flat[key] = np.asarray(w_adj)
            flat[gen_a_key] = np.asarray(ga)
            flat[prefix + "gse_gen_b"] = np.asarray(gb)
            flat[prefix + "gse_spec_a"] = np.asarray(sa)
            flat[prefix + "gse_spec_b"] = np.asarray(sb)
        return flax.traverse_util.unflatten_dict(flat, sep="/")
'''


def patch_gemma(path: Path) -> None:
    src = path.read_text()
    if "import openpi.models.gse as gse" in src:
        print(f"[gse] gemma.py already patched: {path}")
        return
    if GEMMA_IMPORT_ANCHOR not in src:
        raise SystemExit("FATAL: lora import anchor not found in gemma.py")
    src = src.replace(GEMMA_IMPORT_ANCHOR, GEMMA_DISPATCHER, 1)
    if GEMMA_VARIANT_OLD not in src:
        raise SystemExit("FATAL: Variant literal anchor not found in gemma.py")
    src = src.replace(GEMMA_VARIANT_OLD, GEMMA_VARIANT_NEW, 1)
    if GEMMA_GET_CONFIG_ANCHOR not in src:
        raise SystemExit("FATAL: get_config raise anchor not found in gemma.py")
    src = src.replace(GEMMA_GET_CONFIG_ANCHOR, GEMMA_GET_CONFIG_INSERT + GEMMA_GET_CONFIG_ANCHOR, 1)
    # Route the call sites through the dispatcher (aliases above are immune).
    src = src.replace("lora.Einsum(", "_gse_einsum(")
    src = src.replace("lora.FeedForward(", "_gse_feedforward(")
    path.write_text(src)
    print(f"[gse] patched gemma.py: {path}")


def patch_pi0_config(path: Path) -> None:
    src = path.read_text()
    if '"gse" in self.paligemma_variant' in src:
        print(f"[gse] pi0_config.py already patched: {path}")
        return
    for old, new in FREEZE_EDITS:
        if old not in src:
            raise SystemExit(f"FATAL: freeze-filter anchor not found in pi0_config.py:\n  {old!r}")
        src = src.replace(old, new, 1)
    path.write_text(src)
    print(f"[gse] patched pi0_config.py: {path}")


def patch_weight_loaders(path: Path) -> None:
    src = path.read_text()
    if WEIGHT_LOADER_SENTINEL in src:
        print(f"[gse] weight_loaders.py already patched: {path}")
        return
    src = src.rstrip() + "\n" + WEIGHT_LOADER_INSERT
    path.write_text(src)
    print(f"[gse] patched weight_loaders.py: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--openpi-root", type=Path, default=Path.home() / "farm-train" / "openpi")
    args = ap.parse_args()

    root = args.openpi_root
    gemma = root / "src/openpi/models/gemma.py"
    pi0_config = root / "src/openpi/models/pi0_config.py"
    weight_loaders = root / "src/openpi/training/weight_loaders.py"
    gse_src = HERE / "openpi_gse.py"
    gse_dst = root / "src/openpi/models/gse.py"
    for p in (gemma, pi0_config, weight_loaders, gse_src):
        if not p.is_file():
            raise SystemExit(f"FATAL: not found: {p}")

    shutil.copyfile(gse_src, gse_dst)
    print(f"[gse] installed {gse_dst}")
    patch_gemma(gemma)
    patch_pi0_config(pi0_config)
    patch_weight_loaders(weight_loaders)

    for p in (gse_dst, gemma, pi0_config, weight_loaders):
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as exc:
            raise SystemExit(f"FATAL: {p} failed to compile after patch:\n{exc}") from exc
    print("[gse] OK: gse.py + gemma.py + pi0_config.py + weight_loaders.py patched and compile cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
