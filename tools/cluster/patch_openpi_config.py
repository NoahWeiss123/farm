"""Patch openpi/src/openpi/training/config.py to register the FARM UF850 config.

Idempotent — running multiple times is a no-op after the first.

Two insertions:

1. A ``LeRobotFarmDataConfig`` class right after ``LeRobotLiberoDataConfig``.
   It's a clone of libero's config with our LeRobot v2.0 column names
   remapped (``observation.images.base`` / ``observation.images.wrist`` /
   ``observation.state`` / ``action`` → libero's internal naming).
   Libero's input/output transforms work as-is because the data shape is
   identical: single-arm 6-joint + gripper, one base camera, one wrist
   camera, language prompt.

2. A ``TrainConfig(name="pi0_fast_farm_uf850", ...)`` entry in the ``_CONFIGS``
   list. LoRA-tuned (paligemma_variant="gemma_2b_lora" + freeze_filter),
   single-GPU sized.

``setup.sh`` runs this script first because the FULL fine-tune config in
``patch_openpi_config_pi05.py`` reuses the ``LeRobotFarmDataConfig`` class
defined here. The ``pi0_fast_farm_uf850`` LoRA TrainConfig it also registers
is an optional low-budget fallback — it stays available but is not what the
sbatch trains. The default training target is ``pi05_farm_uf850`` (full
fine-tune).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Path to openpi's training config. setup.sh exports OPENPI_CONFIG_PY; the
# literal default is the canonical cluster location for user `nhweiss`.
CFG = Path(os.environ.get(
    "OPENPI_CONFIG_PY",
    "/home/nhweiss/farm-train/openpi/src/openpi/training/config.py",
))
SENTINEL = "FARM_UF850_INSERTED"

DATA_CONFIG_INSERT = '''

# ── FARM_UF850_INSERTED — single-arm 7-DoF bottle manipulation dataset ─
@dataclasses.dataclass(frozen=True)
class LeRobotFarmDataConfig(DataConfigFactory):
    """FARM UF850 dataset config — identical in shape to LeRobot Libero
    (single-arm 7-DoF, 1 base cam + 1 wrist cam, language-conditioned),
    so we reuse libero's input/output transforms verbatim and only swap
    the dataset-side column names in the RepackTransform.
    """

    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Map our LeRobot v2.0 columns → libero's expected internal keys.
        # Libero's LiberoInputs reads ``observation/image``, ``observation/wrist_image``,
        # ``observation/state``, ``actions``, ``prompt``. Our LeRobotDataset
        # exposes ``observation.images.{base,wrist}``, ``observation.state``,
        # ``action``, plus ``prompt`` injected by ``prompt_from_task=True``.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image":        "observation.images.base",
                        "observation/wrist_image":  "observation.images.wrist",
                        "observation/state":        "observation.state",
                        "actions":                  "action",
                        "prompt":                   "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # Our recorded actions are absolute joint positions (action[t] =
        # state[t+1]) but pi0 trains on delta actions for the joints and
        # absolute for the gripper. Mask says: first 6 dims (joints) →
        # delta, last 1 dim (gripper) → absolute.
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Our LeRobot column is "action" (singular, the LeRobot standard).
            # openpi builds delta_timestamps from action_sequence_keys against
            # the RAW dataset column (before the repack renames it to
            # "actions"), and the DataConfig default is ("actions",) — which
            # doesn't exist here. Point it at our real column so action-chunk
            # loading works; the repack still exposes it to the model as
            # "actions". (Mirrors openpi's own ("action",)+repack pattern.)
            action_sequence_keys=("action",),
        )


'''

TRAIN_CONFIG_INSERT = '''    #
    # FARM_UF850_INSERTED — CS153 final project pi0_fast LoRA fine-tune
    #
    # Single 1× H100 80GB on the CS153 cluster (shared, polite default).
    # action_horizon=10 matches the canonical pi0_fast single-arm recipe;
    # at 30 fps that's a 333ms lookahead per chunk.
    TrainConfig(
        name="pi0_fast_farm_uf850",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",   # LoRA on the language tower
        ),
        data=LeRobotFarmDataConfig(
            repo_id="NoahWeiss/farm_uf850_bottle",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        # Freeze the non-LoRA params. Must use the same model variant kwargs
        # here as in ``model=`` above so the freeze filter matches the
        # actually-instantiated parameter tree.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        # EMA off for LoRA per openpi's libero LoRA template.
        ema_decay=None,
        batch_size=64,
        num_train_steps=10_000,
        # 59,183 frames / global batch 64 ≈ 925 steps per epoch → 10k
        # steps ≈ 11 epochs, a reasonable LoRA budget. LoRA fits on a
        # single H100; this config exists as a low-budget fallback to the
        # default pi05_farm_uf850 full fine-tune.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=9_500,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        save_interval=1_000,
        keep_period=2_000,
        num_workers=4,
    ),
'''


def main() -> int:
    text = CFG.read_text()
    if SENTINEL in text:
        print("config.py already patched — no-op")
        return 0

    # Insertion 1: data config class — go right after LeRobotLiberoDataConfig.
    # Find the closing of that class (line starting with the next `@dataclasses.dataclass`
    # after LeRobotLiberoDataConfig's definition).
    libero_idx = text.find("class LeRobotLiberoDataConfig(DataConfigFactory):")
    if libero_idx < 0:
        print("ERROR: couldn't find LeRobotLiberoDataConfig anchor", file=sys.stderr)
        return 1
    # Find the next class def after libero's start
    next_class = text.find("@dataclasses.dataclass(frozen=True)", libero_idx + 1)
    if next_class < 0:
        print("ERROR: couldn't find insertion point after LeRobotLiberoDataConfig", file=sys.stderr)
        return 1
    text = text[:next_class] + DATA_CONFIG_INSERT + text[next_class:]

    # Insertion 2: TrainConfig entry — place inside _CONFIGS list right
    # BEFORE the pi05_libero TrainConfig (which starts right after the
    # pi0_fast_libero_low_mem_finetune entry we want to follow).
    anchor = '    TrainConfig(\n        name="pi05_libero"'
    idx = text.find(anchor)
    if idx < 0:
        print("ERROR: couldn't find pi05_libero anchor for TrainConfig insertion", file=sys.stderr)
        return 1
    text = text[:idx] + TRAIN_CONFIG_INSERT + text[idx:]

    # We also need to import libero_policy in this file (it might already
    # be imported — check).
    if "from openpi.policies import libero_policy" not in text:
        # Add after the existing transform imports near the top.
        import_anchor = "from openpi.transforms"
        ipos = text.find(import_anchor)
        if ipos >= 0:
            line_end = text.find("\n", ipos) + 1
            text = text[:line_end] + "from openpi.policies import libero_policy\n" + text[line_end:]

    CFG.write_text(text)
    print(f"patched {CFG}")
    print("  + LeRobotFarmDataConfig class")
    print("  + TrainConfig(name='pi0_fast_farm_uf850')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
