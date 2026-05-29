"""Direct unit test of the RTC guided sampler — no websocket, no transforms, no
RNG confound. Loads the trained pi0.5 policy and calls sample_actions with FIXED
noise, then asks: does guiding the sample toward a known previous chunk `A`
(full hard-freeze over the whole horizon, max guidance) actually pull the output
onto `A`?

  A   = sample_actions(noiseA)                         # plain reference chunk
  Bp  = sample_actions(noiseB)                         # plain, *different* noise
  Bg  = sample_actions(noiseB, prev=A, freeze=ALL)     # guided toward A

If RTC works: Bg ≈ A (guidance overrides the noise), while Bp deviates from A by
the model's sampling variance. Tests both the eager model method and the jitted
policy._sample_actions (exactly what Policy.infer calls).

Run on a GPU:
  srun --gres=gpu:1 --container-image=... uv run python unit_test.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.training import config as _config

CKPT = "/home/nhweiss/farm-train/openpi/checkpoints/pi05_farm_uf850/farm_uf850_pi05_113/19999"
TASK = "Picking up the bottle and placing it on the box"


def main():
    cfg = _config.get_config("pi05_farm_uf850")
    policy = policy_config.create_trained_policy(cfg, CKPT)
    model = policy._model
    H, D = int(model.action_horizon), int(model.action_dim)
    print(f"loaded pi05_farm_uf850 · action_horizon={H} action_dim={D}")

    obs_dict = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros((7,), np.float32),
        "prompt": TASK,
    }
    inputs = policy._input_transform(dict(obs_dict))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    observation = _model.Observation.from_dict(inputs)

    rng = jax.random.key(0)
    noiseA = jax.random.normal(jax.random.key(1), (1, H, D))
    noiseB = jax.random.normal(jax.random.key(2), (1, H, D))

    def dev(x, y):  # per-step L2 over the 6 joints (normalized space)
        return np.asarray(jnp.linalg.norm((x - y)[0, :, :6], axis=-1))

    # ---- eager (unjitted) model method ----------------------------------
    A = model.sample_actions(rng, observation, noise=noiseA)
    Bp = model.sample_actions(rng, observation, noise=noiseB)
    Bg = model.sample_actions(
        rng, observation, noise=noiseB,
        prev_action_chunk=A,
        inference_delay=jnp.asarray(H, jnp.int32),
        prefix_attention_horizon=jnp.asarray(H, jnp.int32),
        max_guidance_weight=10.0,
    )
    print("\n[eager model.sample_actions] per-step deviation from A (normalized L2):")
    print("  Bp (plain, diff noise) :", np.round(dev(Bp, A), 3))
    print("  Bg (guided→A, freeze ALL):", np.round(dev(Bg, A), 3))
    print(f"  mean: plain {dev(Bp, A).mean():.3f}  guided {dev(Bg, A).mean():.3f}"
          f"   → guidance cuts deviation {100*(1-dev(Bg,A).mean()/max(1e-9,dev(Bp,A).mean())):.0f}%")

    # ---- jitted policy._sample_actions (what Policy.infer calls) ---------
    js = policy._sample_actions
    Aj = js(rng, observation, noise=noiseA)
    Bpj = js(rng, observation, noise=noiseB)
    Bgj = js(rng, observation, noise=noiseB,
             prev_action_chunk=Aj,
             inference_delay=jnp.asarray(H, jnp.int32),
             prefix_attention_horizon=jnp.asarray(H, jnp.int32))
    print("\n[jitted policy._sample_actions] per-step deviation from A (normalized L2):")
    print("  Bp (plain, diff noise) :", np.round(dev(Bpj, Aj), 3))
    print("  Bg (guided→A, freeze ALL):", np.round(dev(Bgj, Aj), 3))
    print(f"  mean: plain {dev(Bpj, Aj).mean():.3f}  guided {dev(Bgj, Aj).mean():.3f}")

    print(f"\nRESULT(sample_actions): guidance reduces deviation "
          f"{100*(1-dev(Bg,A).mean()/max(1e-9,dev(Bp,A).mean())):.0f}% — math "
          f"{'OK' if dev(Bg,A).mean() < 0.6*dev(Bp,A).mean() else 'BROKEN'}")

    # ---- Policy.infer() path (the served path) with kwarg logging --------
    print("\n=== Policy.infer() RTC path (instrumented) ===")
    orig_sa = policy._sample_actions

    def logged_sa(*a, **k):
        shown = {kk: (tuple(vv.shape) if hasattr(vv, "shape") else vv) for kk, vv in k.items()}
        print("   infer → _sample_actions kwargs:", shown)
        return orig_sa(*a, **k)

    policy._sample_actions = logged_sa
    oimg = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros((7,), np.float32),
        "prompt": TASK,
    }
    Ai = policy.infer({**oimg, "rtc_reset": True})["actions"]
    print("   _rtc_prev after A:", None if policy._rtc_prev is None else tuple(policy._rtc_prev.shape))
    Bi = policy.infer({**oimg, "rtc_offset": 0, "rtc_delay": 10})["actions"]
    Bpi = policy.infer({**oimg, "rtc_reset": True})["actions"]
    Ai, Bi, Bpi = np.asarray(Ai), np.asarray(Bi), np.asarray(Bpi)
    da = np.degrees(np.linalg.norm((Bi - Ai)[:, :6], axis=1)).mean()
    dp = np.degrees(np.linalg.norm((Bpi - Ai)[:, :6], axis=1)).mean()
    print(f"   infer B_rtc dev from A: {da:.3f}°   B_plain dev from A: {dp:.3f}°   "
          f"(reduction {100*(1-da/max(1e-9,dp)):.0f}%)")


if __name__ == "__main__":
    main()
