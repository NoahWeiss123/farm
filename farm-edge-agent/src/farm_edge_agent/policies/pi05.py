"""Pi0.5 policy client — runs inference on a remote Modal endpoint.

The π0.5 model itself (https://github.com/Physical-Intelligence/openpi)
needs ~24 GB of GPU memory and a JAX/Flax stack, so it can't run on
local M-series Macs. We serve it on Modal (see
``farm-cloud/modal/pi05_serve.py`` for the deployable function) and call
into it from the edge daemon via HTTP.

Observation/action wire format matches openpi's ``DroidInputs``:

  request body:
    {
      "observation": {
        "exterior_image_1_left": <base64 png>,
        "wrist_image_left":      <base64 png>,
        "joint_position":        [j0, ..., j6],      # 7 dims
        "gripper_position":      [g]                 # 0..1
      },
      "prompt": "pick the red block ..."
    }

  response:
    {
      "actions":         [[d0, ..., d7], ...],      # (horizon, 8)
      "control_period_s": 0.05
    }

Action semantics:
  - dims 0..6 = joint deltas (radians) to add to current joint_position
  - dim 7    = absolute gripper target on [0, 1] (0 open, 1 closed)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger("farm.policies.pi05")

DEFAULT_ENDPOINT = os.environ.get("FARM_PI05_ENDPOINT", "")
DEFAULT_TIMEOUT = float(os.environ.get("FARM_PI05_TIMEOUT_S", "30"))


@dataclass
class Pi05Result:
    actions: np.ndarray            # shape (horizon, 8)
    control_period_s: float
    latency_s: float


def _png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class Pi05Policy:
    """HTTP client for the Modal-hosted π0.5 inference function.

    ``endpoint`` is the full URL of the Modal app's ``/infer`` route. When
    unset (which is the default in local dev), ``infer`` raises so the
    supervisor can fall back to the GPT-skill pipeline rather than
    silently producing zero actions.
    """

    endpoint: str = DEFAULT_ENDPOINT
    timeout_s: float = DEFAULT_TIMEOUT
    auth_header: str | None = None

    def configured(self) -> bool:
        return bool(self.endpoint)

    def infer(self, observation: dict, prompt: str) -> Pi05Result:
        if not self.configured():
            raise RuntimeError(
                "Pi05Policy.endpoint is not set — deploy the Modal app at "
                "farm-cloud/modal/pi05_serve.py and export "
                "FARM_PI05_ENDPOINT=https://<your-modal-app>.modal.run/infer"
            )
        body = {
            "observation": {
                "exterior_image_1_left": _png_b64(
                    observation["observation/exterior_image_1_left"]
                ),
                "wrist_image_left": _png_b64(
                    observation["observation/wrist_image_left"]
                ),
                "joint_position": np.asarray(
                    observation["observation/joint_position"], dtype=np.float32
                ).tolist(),
                "gripper_position": np.asarray(
                    observation["observation/gripper_position"], dtype=np.float32
                ).tolist(),
            },
            "prompt": prompt,
        }
        headers = {"content-type": "application/json"}
        if self.auth_header:
            headers["authorization"] = self.auth_header
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"pi05 endpoint returned {e.code}: {e.read()[:200]!r}"
            ) from e
        latency = time.perf_counter() - t0
        actions = np.asarray(payload["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[-1] < 7:
            raise RuntimeError(
                f"pi05 response has {actions.shape[-1]} action dims; expected ≥7"
            )
        return Pi05Result(
            actions=actions,
            control_period_s=float(payload.get("control_period_s", 0.05)),
            latency_s=latency,
        )


def run_pi05_loop(
    driver: Any,
    policy: Pi05Policy,
    prompt: str,
    *,
    max_steps: int = 400,
    chunks_per_call: int = 10,
    is_delta: bool = True,
    on_step: Callable[[dict], None] | None = None,
    stop_event: Any = None,
) -> dict:
    """Closed-loop π0.5 control. Calls the policy, executes the returned
    action chunk on the driver, observes, repeats.

    The model returns an action *chunk* (horizon × 8); we execute the
    first ``chunks_per_call`` actions before asking the model again. This
    is the standard openpi pattern — amortizes inference latency.
    """
    if not policy.configured():
        raise RuntimeError("Pi05Policy not configured; see Pi05Policy.infer")
    metrics = {"calls": 0, "actions": 0, "latency_total_s": 0.0}
    elapsed = 0
    while elapsed < max_steps:
        if stop_event is not None and stop_event.is_set():
            break
        obs = driver.pi05_observation()
        result = policy.infer(obs, prompt)
        metrics["calls"] += 1
        metrics["latency_total_s"] += result.latency_s
        if on_step is not None:
            on_step({"type": "pi05_infer", "latency_s": result.latency_s,
                     "chunk_len": int(result.actions.shape[0])})
        # Step ``chunks_per_call`` actions from this chunk before re-querying.
        n = min(chunks_per_call, result.actions.shape[0])
        for i in range(n):
            if stop_event is not None and stop_event.is_set():
                break
            a = result.actions[i]
            grip = float(a[7]) if a.shape[0] >= 8 else None
            driver.apply_joint_action(
                a[:7],
                is_delta=is_delta,
                gripper_target=grip,
                steps=int(round(result.control_period_s / driver._model.opt.timestep)),
            )
            metrics["actions"] += 1
            elapsed += 1
            if elapsed >= max_steps:
                break
    return metrics


__all__ = ["Pi05Policy", "Pi05Result", "run_pi05_loop"]
