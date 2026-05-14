"""Modal serving function for π0.5 inference.

Deploy:
    pip install modal && modal token new
    modal deploy farm-cloud/modal/pi05_serve.py

The deploy command prints a URL like
    https://<your-workspace>--farm-pi05-infer.modal.run
which you point the edge daemon at:
    export FARM_PI05_ENDPOINT=https://<your-workspace>--farm-pi05-infer.modal.run
    farm serve

Request body shape (matches openpi DROID policy):

    {
      "observation": {
        "exterior_image_1_left": <base64 PNG>,
        "wrist_image_left":      <base64 PNG>,
        "joint_position":        [j0..j6],
        "gripper_position":      [g]
      },
      "prompt": "pick the red block ..."
    }

Response:
    {
      "actions":          [[Δq0..Δq6, g_target], ...],   # shape (horizon, 8)
      "control_period_s": 0.05
    }
"""

from __future__ import annotations

import modal

CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "fastapi[standard]",
        "pillow",
        "numpy",
        "einops",
    )
    # openpi is installed from source — Physical Intelligence ships the
    # repo without a PyPI release. Pinning to main is OK for a private
    # demo; pin a commit for prod.
    .run_commands(
        "git clone --depth=1 https://github.com/Physical-Intelligence/openpi /opt/openpi",
        "pip install -e /opt/openpi",
        "pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
)

app = modal.App("farm-pi05", image=image)
volume = modal.Volume.from_name("farm-pi05-cache", create_if_missing=True)


@app.cls(
    gpu="A100-40GB",          # π0.5 needs 24 GB+, A100-40GB is comfortable
    timeout=600,
    container_idle_timeout=300,
    volumes={"/root/openpi_cache": volume},
)
class Pi05Server:
    @modal.enter()
    def load(self) -> None:
        """Cold-start: download checkpoint + warm up the policy."""
        import os
        os.environ.setdefault("OPENPI_CACHE", "/root/openpi_cache")
        from openpi.training import config as _config
        from openpi.policies import policy_config
        from openpi.shared import download

        ckpt = download.maybe_download(CHECKPOINT)
        cfg = _config.get_config("pi05_droid")
        self.policy = policy_config.create_trained_policy(cfg, ckpt)
        # Warm the JAX graph with a dummy obs so first real request is fast.
        from openpi.policies import droid_policy
        _ = self.policy.infer(droid_policy.make_droid_example())

    @modal.fastapi_endpoint(method="POST")
    def infer(self, body: dict) -> dict:
        import base64
        import io

        import numpy as np
        from PIL import Image

        obs = body["observation"]

        def png_decode(b64: str) -> np.ndarray:
            data = base64.b64decode(b64)
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

        example = {
            "observation/exterior_image_1_left": png_decode(
                obs["exterior_image_1_left"]
            ),
            "observation/wrist_image_left": png_decode(obs["wrist_image_left"]),
            "observation/joint_position": np.asarray(
                obs["joint_position"], dtype=np.float32
            ),
            "observation/gripper_position": np.asarray(
                obs["gripper_position"], dtype=np.float32
            ),
            "prompt": body.get("prompt", ""),
        }
        result = self.policy.infer(example)
        actions = np.asarray(result["actions"])
        return {
            "actions": actions.tolist(),
            "control_period_s": 1.0 / 20.0,   # DROID was trained at 20 Hz
        }


# Standalone runner so you can hit the endpoint locally with `modal serve`.
if __name__ == "__main__":
    with app.run():
        pass
