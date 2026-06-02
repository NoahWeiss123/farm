"""Patch openpi's models/model.py to make train-time image augmentation
env-controlled, and add a strong "heavy" domain-randomization recipe.

WHY
───
The deployed π0.5 fine-tunes fail live mostly because the **deployment scene is
out-of-distribution** for the vision tower: the demos were recorded in one room
(living-room background, one camera pose, one lighting) and the policy is run in
a different room (plywood wall, repositioned/wider camera, a person in frame).
No fine-tune *architecture* fixes a domain shift — but **domain randomization**
(aggressively perturbing image appearance during training) is the standard
training-time mitigation: it forces the encoder to be invariant to lighting,
colour, background and mild viewpoint, so a new room looks less alien.

openpi already augments during training inside ``preprocess_observation`` (when
``train=True``) but with a *moderate* fixed recipe (RandomCrop .95 + Rotate ±5 +
ColorJitter, no hue). This patch:

  * gates that block on ``FARM_AUG_LEVEL`` (``off`` / ``default`` / ``heavy``),
    read once at import; and
  * adds a ``heavy`` recipe: wider brightness/contrast/saturation **+ hue**
    jitter (the living-room→plywood colour/background shift), per-channel gamma
    (colour-temperature shift), occasional grayscale (don't over-rely on
    absolute colour), gaussian blur (defocus), and a slightly stronger
    crop/rotate (camera-reposition).

Backward-compatible: unset ⇒ ``default`` ⇒ byte-identical to stock openpi, so
existing configs and the **serving path** (always ``train=False``) are
unchanged. The training sbatch exports ``FARM_AUG_LEVEL`` per run.

Idempotent (sentinel-guarded). Run from the login pod:

    python3 patch_openpi_aug.py --openpi-root ~/farm-train/openpi
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SENTINEL = "FARM_AUG_PATCH_INSERTED"

HELPER_BLOCK = '''
# ── FARM_AUG_PATCH_INSERTED — env-controlled image domain randomization ────────
# FARM_AUG_LEVEL picks the train-time augmentation strength (read once at import;
# the training sbatch exports it before launching scripts/train.py):
#   "off"     — no augmentation (clean baseline / ablation)
#   "default" — openpi's stock recipe (unchanged behaviour; this is the default)
#   "heavy"   — strong domain randomization for cross-room/-lighting transfer
# Unset ⇒ "default", so the serving path and every existing config are unchanged.
_FARM_AUG_LEVEL = os.environ.get("FARM_AUG_LEVEL", "default").strip().lower()


def _farm_aug_transforms(key: str, height: int, width: int) -> list:
    """augmax transforms for one camera under the active FARM_AUG_LEVEL.

    Geometric ops (crop/rotate) hit the BASE camera only — never the wrist,
    whose tight top-down view carries fine grasp geometry. Photometric ops hit
    every camera (lighting / background / appearance is what shifts across
    rooms). Returns [] for "off" (the caller then skips the augmax chain).
    """
    if _FARM_AUG_LEVEL == "off":
        return []
    is_base = "wrist" not in key
    if _FARM_AUG_LEVEL == "heavy":
        # ORDER MATTERS for numerical safety: gamma does x**g, which is NaN for
        # x<0 and non-integer g. So gamma runs FIRST, on the clean [0,1] image
        # (geometric crop/resize/rotate use bilinear interp = convex → stays in
        # [0,1], so they're safe to precede it). Photometric ops that can
        # overshoot [0,1] (ColorJitter) come AFTER gamma; the caller clips the
        # chain output back to [0,1] before the [-1,1] rescale.
        t = []
        if is_base:
            t += [
                augmax.RandomCrop(int(width * 0.90), int(height * 0.90)),
                augmax.Resize(width, height),
                augmax.Rotate((-8, 8)),
            ]
        t += [
            augmax.RandomChannelGamma(range=(0.7, 1.5), p=0.5),  # colour-temperature shift (safe input)
            # Wide photometric jitter INCLUDING hue — the dominant lever for a
            # living-room→plywood-room appearance / background / lighting shift.
            augmax.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.08, p=0.9),
            augmax.RandomGrayscale(p=0.20),                      # don't lean on absolute colour
            augmax.GaussianBlur(sigma=2, p=0.30),                # defocus / motion robustness
        ]
        return t
    # "default" (or any unrecognised value) → openpi's stock recipe, unchanged.
    t = []
    if is_base:
        t += [
            augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
            augmax.Resize(width, height),
            augmax.Rotate((-5, 5)),
        ]
    t += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
    return t


'''

# The exact stock block we replace, matched by its first and last lines (span
# replacement is robust to small edits of the lines in between).
OLD_START = "            transforms = []\n"
OLD_END = "            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)\n"

NEW_BLOCK = (
    "            height, width = image.shape[1:3]\n"
    "            transforms = _farm_aug_transforms(key, height, width)\n"
    "            if transforms:\n"
    "                sub_rngs = jax.random.split(rng, image.shape[0])\n"
    "                image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)\n"
    "                image = jnp.clip(image, 0.0, 1.0)  # guard any overshoot before [-1,1] rescale\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi-root", default="/home/nhweiss/farm-train/openpi")
    args = ap.parse_args()

    model_py = Path(args.openpi_root) / "src/openpi/models/model.py"
    if not model_py.is_file():
        print(f"ERROR: {model_py} not found", file=sys.stderr)
        return 1

    text = model_py.read_text()
    if SENTINEL in text:
        print("model.py already patched for FARM aug — no-op")
        return 0

    # 1) ensure `import os` (the helper reads os.environ).
    if not re.search(r"^import os$", text, re.M):
        # insert right after the first `import augmax` (guaranteed present here).
        text = text.replace("import augmax\n", "import augmax\nimport os\n", 1)

    # 2) insert the helper + level constant right before preprocess_observation.
    anchor = "def preprocess_observation("
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find preprocess_observation", file=sys.stderr)
        return 1
    text = text[:idx] + HELPER_BLOCK.lstrip("\n") + text[idx:]

    # 3) span-replace the stock augmentation block (scoped to the function body
    #    so a stray "transforms = []" elsewhere can't be matched by mistake).
    fn_idx = text.find(anchor)
    i = text.find(OLD_START, fn_idx)
    j = text.find(OLD_END, fn_idx)
    if i < 0 or j < 0 or j < i:
        print("ERROR: couldn't locate the stock augmentation block to replace.\n"
              f"       start_found={i >= 0} end_found={j >= 0}", file=sys.stderr)
        return 1
    j_end = j + len(OLD_END)
    text = text[:i] + NEW_BLOCK + text[j_end:]

    model_py.write_text(text)
    print(f"patched {model_py}")
    print("  + FARM_AUG_LEVEL gate (off/default/heavy) + _farm_aug_transforms()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
