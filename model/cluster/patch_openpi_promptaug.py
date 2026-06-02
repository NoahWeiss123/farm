"""Wire env-gated prompt-paraphrase augmentation into openpi.

Two actions, both idempotent:

  1. Copy ``farm_prompt_aug.py`` (staged beside this script) into the openpi
     source tree at ``src/openpi/farm_prompt_aug.py`` so it imports as
     ``openpi.farm_prompt_aug``.
  2. Patch ``src/openpi/training/config.py`` to (a) import that module and
     (b) prepend ``farm_prompt_aug.PromptParaphrase()`` to the
     ``data_transforms`` inputs of ``LeRobotFarmDataConfig`` ONLY (scoped to
     that class, so the stock libero config is untouched).

The transform is identity unless ``FARM_PROMPT_AUG`` is set, so serve/eval are
unaffected; the training sbatch sets the var for the runs that want paraphrasing.

Run from the login pod (setup order: after patch_openpi_config.py):

    python3 patch_openpi_promptaug.py --openpi-root ~/farm-train/openpi
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SENTINEL = "FARM_PROMPT_AUG_INSERTED"
IMPORT_LINE = "from openpi import farm_prompt_aug  # FARM_PROMPT_AUG_INSERTED\n"

# Scoped to LeRobotFarmDataConfig (the libero clone uses the identical line, so
# we must search AFTER the Farm class definition, not from the top of the file).
ANCHOR = "inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],"
REPLACEMENT = (
    "inputs=[farm_prompt_aug.PromptParaphrase(), "
    "libero_policy.LiberoInputs(model_type=model_config.model_type)],"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi-root", default="/home/nhweiss/farm-train/openpi")
    args = ap.parse_args()
    root = Path(args.openpi_root)

    # 1) copy the module into the openpi package.
    src_mod = Path(__file__).resolve().parent / "farm_prompt_aug.py"
    if not src_mod.is_file():
        print(f"ERROR: {src_mod} not staged beside this patch", file=sys.stderr)
        return 1
    dst_mod = root / "src/openpi/farm_prompt_aug.py"
    if not dst_mod.parent.is_dir():
        print(f"ERROR: {dst_mod.parent} not found (is --openpi-root right?)", file=sys.stderr)
        return 1
    shutil.copyfile(src_mod, dst_mod)
    print(f"copied → {dst_mod}")

    # 2) patch config.py.
    cfg = root / "src/openpi/training/config.py"
    text = cfg.read_text()
    if SENTINEL in text:
        print("config.py already wired for prompt-aug — no-op")
        return 0
    if "class LeRobotFarmDataConfig" not in text:
        print("ERROR: LeRobotFarmDataConfig not in config.py (run patch_openpi_config.py first)",
              file=sys.stderr)
        return 1

    # 2a) import, placed after the libero_policy import the data config relies on.
    lib_import = "from openpi.policies import libero_policy\n"
    if lib_import in text:
        text = text.replace(lib_import, lib_import + IMPORT_LINE, 1)
    else:  # fall back to inserting after the first openpi import
        anchor_imp = "import openpi.transforms as _transforms\n"
        text = text.replace(anchor_imp, anchor_imp + IMPORT_LINE, 1)

    # 2b) prepend PromptParaphrase to the Farm config's data_transforms inputs,
    #     scoped to the LeRobotFarmDataConfig class body.
    farm_idx = text.find("class LeRobotFarmDataConfig")
    a_idx = text.find(ANCHOR, farm_idx)
    if farm_idx < 0 or a_idx < 0:
        print("ERROR: couldn't find LiberoInputs anchor inside LeRobotFarmDataConfig",
              file=sys.stderr)
        return 1
    text = text[:a_idx] + REPLACEMENT + text[a_idx + len(ANCHOR):]

    cfg.write_text(text)
    print(f"patched {cfg}")
    print("  + import openpi.farm_prompt_aug")
    print("  + PromptParaphrase() prepended to LeRobotFarmDataConfig.data_transforms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
