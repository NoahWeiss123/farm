#!/usr/bin/env python3
"""Make openpi's train loop log its per-step metrics through ``logging``.

openpi prints the metrics line with ``pbar.write(f"Step {step}: {info_str}")``.
That goes to the tqdm output stream, which openpi redirects to a logging bridge
that only forwards the progress bar — so the ``Step N: loss=…`` line never
reaches stdout (and with ``--no-wandb-enabled`` the loss has nowhere else to
go). The /train dashboard parses the SLURM log, so it can show progress (from
the tqdm line) but never a loss curve.

Fix: emit the same line via ``logging.info`` right after ``pbar.write``. The
logging handler writes straight to stdout, so the loss lands in the SLURM log
and ``server/cluster.parse_log`` can plot it. The console output is unchanged.

Idempotent (sentinel-guarded) and py_compile-checked, like the other patches.

    python3 patch_openpi_train_logging.py --openpi-root ~/farm-train/openpi
"""
from __future__ import annotations

import argparse
import os
import py_compile
import re
import sys

SENTINEL = "logging.info(f\"Step {step}: {info_str}\")"
# Match the existing pbar.write line and capture its leading indentation.
PBAR_RE = re.compile(r'^(\s*)pbar\.write\(f"Step \{step\}: \{info_str\}"\)\s*$', re.M)


def patch(train_py: str) -> bool:
    """Insert the logging.info mirror. Returns True if a write happened."""
    with open(train_py, encoding="utf-8") as f:
        src = f.read()
    if SENTINEL in src:
        print(f"    already patched: {train_py}")
        return False
    m = PBAR_RE.search(src)
    if m is None:
        print(
            f"ERROR: could not find the pbar.write metrics line in {train_py}\n"
            "       (openpi's train loop may have changed — patch by hand)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    indent = m.group(1)
    insert = f"{m.group(0)}\n{indent}{SENTINEL}"
    src = src[: m.start()] + insert + src[m.end():]
    with open(train_py, "w", encoding="utf-8") as f:
        f.write(src)
    py_compile.compile(train_py, doraise=True)
    print(f"    patched + compiles: {train_py}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--openpi-root",
        default=os.path.expanduser("~/farm-train/openpi"),
        help="openpi checkout root (default: ~/farm-train/openpi)",
    )
    args = ap.parse_args()
    train_py = os.path.join(args.openpi_root, "scripts", "train.py")
    if not os.path.isfile(train_py):
        print(f"ERROR: {train_py} not found", file=sys.stderr)
        raise SystemExit(1)
    patch(train_py)


if __name__ == "__main__":
    main()
