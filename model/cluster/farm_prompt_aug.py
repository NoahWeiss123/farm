"""Env-gated prompt-paraphrase augmentation for the FARM UF850 π0.5 fine-tunes.

Copied into ``openpi/src/openpi/farm_prompt_aug.py`` by ``patch_openpi_promptaug.py``
and wired in as the first ``data_transforms`` input of ``LeRobotFarmDataConfig``.

WHY
───
The dataset has exactly **two fixed task strings**, so π0.5's language pathway
overfits to those literal strings and any novel phrasing is out-of-distribution
(FINDINGS.md cause #3). This is doubly damaging here because the two tasks are
visually ambiguous at their endpoints — only the *prompt* distinguishes "place
on the box" from "move to the desk". Sampling a paraphrase per training example
keeps the language conditioning alive (and robust to rephrasing) without
changing task semantics.

SAFETY
──────
Active ONLY when ``FARM_PROMPT_AUG`` is truthy in the environment (the training
sbatch sets it). At serve / eval time the var is unset, so ``__call__`` is a
strict identity and the canonical prompt passes through byte-for-byte. Unknown
prompts also pass through unchanged.
"""
from __future__ import annotations

import dataclasses
import os
import random

import openpi.transforms as _transforms

# Exact canonical task strings. The first two are the original dataset3 bottle
# tasks (kept for backward compat with old-dataset runs); the latter four are the
# dataset4 multi-object tasks (bottle/bear/hat/duck → box). Keys MUST match the
# meta "description" byte-for-byte (prompt_from_task=True feeds it straight in).
_T0 = "Picking up the bottle and placing it on the box"
_T1 = "Picking up the bottle off of the box and putting it on the desk"
_BEAR = "Pick up the stuffed bear and place it on the box"
_HAT = "Pick up the hat and place it on the box"
_DUCK = "Pick up the rubber duck and place it on the box"

# Paraphrase pools — semantics fixed, phrasing/verbs varied. The canonical
# string is included so it is still seen during training.
_PARAPHRASES: dict[str, list[str]] = {
    _T0: [
        _T0,
        "pick up the bottle and place it on the box",
        "put the bottle on the box",
        "place the bottle onto the box",
        "grab the bottle and set it on the box",
        "lift the bottle and put it on top of the box",
        "move the bottle onto the box",
        "set the bottle down on the box",
    ],
    _T1: [
        _T1,
        "pick up the bottle from the box and put it on the desk",
        "take the bottle off the box and place it on the desk",
        "move the bottle from the box to the desk",
        "grab the bottle off the box and set it on the desk",
        "lift the bottle off the box and put it on the desk",
        "put the bottle on the desk",
        "remove the bottle from the box and place it on the desk",
    ],
    _BEAR: [
        _BEAR,
        "pick up the bear and place it on the box",
        "put the stuffed bear on the box",
        "place the teddy bear onto the box",
        "grab the stuffed bear and set it on the box",
        "lift the bear and put it on top of the box",
        "move the stuffed bear onto the box",
        "set the bear down on the box",
    ],
    _HAT: [
        _HAT,
        "pick up the hat and place it on the box",
        "put the hat on the box",
        "place the hat onto the box",
        "grab the hat and set it on the box",
        "lift the hat and put it on top of the box",
        "move the hat onto the box",
        "set the hat down on the box",
    ],
    _DUCK: [
        _DUCK,
        "pick up the duck and place it on the box",
        "put the rubber duck on the box",
        "place the duck onto the box",
        "grab the rubber duck and set it on the box",
        "lift the rubber duck and put it on top of the box",
        "move the rubber duck onto the box",
        "set the duck down on the box",
    ],
}

_ENV_FLAG = "FARM_PROMPT_AUG"


@dataclasses.dataclass(frozen=True)
class PromptParaphrase(_transforms.DataTransformFn):
    """Replace a known canonical task string with a random paraphrase.

    No-op unless ``os.environ['FARM_PROMPT_AUG']`` is set (training only).
    Unknown prompts pass through unchanged → serve/eval safe.
    """

    def __call__(self, data: dict) -> dict:
        # Off unless explicitly enabled. Treat "0"/"false"/"no"/"off"/"" as off
        # ("0" is a *truthy* Python string, so a plain bool check is wrong here).
        if os.environ.get(_ENV_FLAG, "").strip().lower() in ("", "0", "false", "no", "off"):
            return data
        prompt = data.get("prompt")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8", "ignore")
        pool = _PARAPHRASES.get(prompt) if isinstance(prompt, str) else None
        if not pool:
            return data
        return {**data, "prompt": random.choice(pool)}
