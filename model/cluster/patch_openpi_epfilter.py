"""Patch openpi's data_loader.py for env-gated EPISODE-SUBSET training via
``FARM_EP_RANGE``, used to train a per-task LoRA on one task's episodes of the
multi-object set WITHOUT rebuilding the dataset.

IMPLEMENTATION — why a frame-index Subset, not ``LeRobotDataset(episodes=...)``:
The installed lerobot 0.1.0 does NOT remap ``episode_index`` when ``episodes=``
is passed — ``__getitem__`` reads the ORIGINAL parquet episode_index (e.g. 299 for
the first bear episode) and indexes ``episode_data_index['from']`` (a DENSE array of
length = subset size), so any task whose range does not start at 0 (bear/duck/hat)
throws IndexError on the first batch. Instead we build the FULL dataset and wrap it
in a ``torch.utils.data.Subset`` over the GLOBAL FRAME indices belonging to the
requested episodes (the tasks are contiguous, so this is a contiguous frame slice).
Frame boundaries come from the per-episode lengths in the dataset metadata, so the
original episode indexing is preserved and the action-chunk (delta_timestamps)
loading is byte-identical to full-set training.

``FARM_EP_RANGE`` formats (read at dataset build):
  * ``"299:329"``  — half-open episode range → episodes 299..328
  * ``"0,3,7"``    — explicit episode list
  * unset / empty  — ALL episodes (stock behaviour; byte-identical to upstream)

compute_norm_stats.py also calls create_torch_dataset, so do NOT compute per-task
norm stats — pre-place the FFT base model's full-set norm_stats.json (the LoRA
sbatch does), keeping normalization shared across every task-LoRA.

Backward-compatible (unset ⇒ stock). Idempotent. py_compile-checked.

    python3 patch_openpi_epfilter.py --openpi-root ~/farm-train/openpi
"""
from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

SENTINEL = "FARM_EP_RANGE_INSERTED"

HELPER = '''
# ── FARM_EP_RANGE_INSERTED — env-gated LeRobot episode-subset (frame Subset) ───
# Restrict training to a subset of episodes WITHOUT rebuilding the dataset and
# WITHOUT lerobot's broken episodes= re-indexing: build the full dataset, then wrap
# it in a torch Subset over the global frame indices of the requested episodes.
# "a:b" → episodes range(a,b); "a,b,c" → explicit list; unset/empty → unchanged.
def _farm_episode_subset(dataset, dataset_meta):
    import os
    spec = os.environ.get("FARM_EP_RANGE", "").strip()
    if not spec:
        return dataset
    import numpy as np
    import torch
    if ":" in spec:
        a, b = spec.split(":", 1)
        eps = list(range(int(a), int(b)))
    else:
        eps = [int(x) for x in spec.split(",") if x.strip() != ""]
    # per-episode global frame boundaries from the dataset metadata lengths
    n_ep = int(dataset_meta.total_episodes)
    lengths = [int(dataset_meta.episodes[i]["length"]) for i in range(n_ep)]
    ends = np.cumsum(lengths)
    starts = ends - np.asarray(lengths)
    idxs = []
    for e in eps:
        idxs.extend(range(int(starts[e]), int(ends[e])))
    print(f"[FARM_EP_RANGE] training on {len(eps)} episodes "
          f"({eps[0]}..{eps[-1]}) = {len(idxs)} frames (subset of {len(dataset)})", flush=True)
    return torch.utils.data.Subset(dataset, idxs)


'''

OLD = (
    "    if data_config.prompt_from_task:\n"
    "        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])\n"
)
NEW = (
    "    dataset = _farm_episode_subset(dataset, dataset_meta)\n"
    "\n"
    "    if data_config.prompt_from_task:\n"
    "        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi-root", default="/home/nhweiss/farm-train/openpi")
    args = ap.parse_args()
    dl = Path(args.openpi_root) / "src/openpi/training/data_loader.py"
    if not dl.is_file():
        print(f"ERROR: {dl} not found", file=sys.stderr)
        return 1
    text = dl.read_text()
    if SENTINEL in text:
        print("data_loader.py already patched for FARM_EP_RANGE — no-op")
        return 0
    if OLD not in text:
        print("ERROR: couldn't find the prompt_from_task block to patch.\n"
              "       (openpi data_loader.py changed — patch by hand)", file=sys.stderr)
        return 1
    anchor = "def create_torch_dataset("
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find create_torch_dataset", file=sys.stderr)
        return 1
    text = text[:idx] + HELPER.lstrip("\n") + "\n\n" + text[idx:]
    text = text.replace(OLD, NEW, 1)
    dl.write_text(text)
    try:
        py_compile.compile(str(dl), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"ERROR: data_loader.py failed to compile after patch:\n{exc}", file=sys.stderr)
        return 1
    print(f"patched {dl}")
    print("  + _farm_episode_subset() (frame Subset honouring FARM_EP_RANGE)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
