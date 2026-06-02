"""Make the GSE specialized-expert count configurable via an env var.

The expert-count sweep runs 6 training jobs (and their evals) against a SINGLE
shared openpi checkout. The number of specialized experts is otherwise hardcoded
(``num_specialized=7``) in the ``gemma_2b_gse`` / ``gemma_300m_gse`` configs that
``patch_openpi_gse.py`` writes into ``models/gemma.py`` — so without this patch
all jobs would build the same architecture.

This patch replaces that literal with
``int(os.environ.get("FARM_GSE_NUM_SPECIALIZED", "7"))`` (and ensures ``import
os``). The count is read at model-build time — when ``gemma.get_config(
"gemma_2b_gse")`` runs inside each training/eval process — so each SLURM job's
container env selects its own expert count with no source-file conflict.
``GSESVDWeightLoader`` already derives ``num_specialized`` from the param shapes
(``e = int(spec_a.shape[1])``), so SVD-init adapts automatically. The default
(env unset) stays 7 specialized = 8 total experts (the shipped GSE default), so
this is behaviour-preserving for every existing config.

Idempotent (re-running is a no-op), py_compile-verified. Run AFTER
``patch_openpi_gse.py`` (which installs the GSEConfig sites). ``setup.sh`` order:
gse -> this -> config patches.
"""
from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

OLD = "gse.GSEConfig(generalized_rank=2, num_specialized=7, expert_rank=2)"
NEW = (
    "gse.GSEConfig(generalized_rank=2, "
    'num_specialized=int(os.environ.get("FARM_GSE_NUM_SPECIALIZED", "7")), '
    "expert_rank=2)"
)
SENTINEL = "FARM_GSE_NUM_SPECIALIZED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--openpi-root", type=Path, default=Path.home() / "farm-train" / "openpi")
    args = ap.parse_args()

    gemma = args.openpi_root / "src/openpi/models/gemma.py"
    if not gemma.is_file():
        print(f"FATAL: not found: {gemma}", file=sys.stderr)
        return 1

    src = gemma.read_text()
    if SENTINEL in src:
        print("[experts-env] gemma.py already env-parameterized — no-op")
        return 0
    if OLD not in src:
        print(
            f"FATAL: GSEConfig anchor not found in {gemma}.\n"
            f"       Expected literal: {OLD}\n"
            "       Run patch_openpi_gse.py first.",
            file=sys.stderr,
        )
        return 1

    n = src.count(OLD)
    src = src.replace(OLD, NEW)

    # Ensure `import os` is present (gemma.py does NOT import it by default).
    if "import os\n" not in src:
        if "import dataclasses\n" in src:
            src = src.replace("import dataclasses\n", "import dataclasses\nimport os\n", 1)
        else:
            src = "import os\n" + src

    gemma.write_text(src)
    try:
        py_compile.compile(str(gemma), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"FATAL: gemma.py failed to compile after patch:\n{exc}", file=sys.stderr)
        return 1

    print(
        f"[experts-env] patched {gemma}: {n} GSEConfig site(s) now read "
        "FARM_GSE_NUM_SPECIALIZED (default 7); import os ensured; compiles clean"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
