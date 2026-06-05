#!/usr/bin/env python3
"""Serve π0.5 with **live per-object LoRA hot-swap** on a resident FFT-56k base.

The /user page breaks a task into per-object steps and drives them by changing
the policy *prompt* between steps. This server keeps the heavy full-fine-tune
(FFT-56k) base resident on the GPU and, per inference, looks at the incoming
prompt, maps it to a skill, and — if the skill changed — swaps **only the small
``lora_a``/``lora_b`` adapter leaves**, leaving the multi-GB base in place. That
is the thing the user asked for: "swap the LoRAs without re-getting the entire
FFT-56k, just make it smart."

Every ``NoahWeiss/farm_fftlora_<task>`` checkpoint stores the SAME frozen FFT-56k
dense base plus that task's adapter, so we build one working policy from a single
task checkpoint (base + one adapter resident) and preload just the ``lora_*``
leaves of every other task.

HOW THE SWAP WORKS (and why it isn't a naive attribute set)
-----------------------------------------------------------
openpi's ``Policy`` is **nnx-based**: it holds an ``nnx.Module`` in ``self._model``
and a jitted sampler in ``self._sample_actions`` that closes over the model's nnx
state — there is no plain ``self._params`` pytree to overwrite (confirmed against
this repo's own ``patch_openpi_rtc.py``, whose ``Policy.infer`` calls
``self._sample_actions(rng, observation, **kwargs)`` with no params argument). So
swapping means: mutate the ``lora_a``/``lora_b`` leaves inside the model's nnx
state (selected by the same ``.*lora.*`` path the freeze filter uses), push them
back with ``nnx.update``, and **rebuild ``self._sample_actions``** so the next
inference samples with the new adapter. The FFT-56k base leaves are never touched
or reloaded — only the adapter arrays change.

Cost: rebuilding the jitted sampler triggers a JAX trace/compile on the first
inference after a swap (seconds for this model), NOT a multi-GB base reload. JAX
caches by shape, so once each skill has run once, repeat swaps reuse the base
state in memory; the adapter arrays are pre-staged on-device at startup.

‼ MUST VALIDATE ON A GPU before relying on it. This file cannot be exercised off
the cluster (no openpi / no GPU here). On the first ``sbatch
serve_fftlora_hotswap.sbatch`` run, confirm the log shows:
    >>> hot-swap armed · <N> adapted leaves · skills=[...]
and, when the /user prompt changes object, a line like:
    swap skill bottle -> bear  (<k> leaves, <ms> ms)
If instead you see "HOT-SWAP DISABLED", the nnx introspection didn't match this
openpi revision — the server still serves the base safely, and the warning names
what to adjust (likely the State flatten/update API or the lora path match).

Run inside the openpi NGC container (see serve_fftlora_hotswap.sbatch):

  uv run python serve_policy_hotswap.py \
      --config pi05_fftlora \
      --policy-dir <a task ckpt dir, e.g. fftlora_bottle/11999> \
      --adapter bottle=<ckpt> --adapter bear=<ckpt> \
      --adapter duck=<ckpt>  --adapter hat=<ckpt> \
      --default-skill bottle --port 8000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serve.hotswap")

# Sentinel "skill" meaning NO adapter: zero the LoRA leaves so the bare FFT-56k
# base (the generalist) runs, for objects no trained skill fits.
BASE_SKILL = "__base__"

# Prompt → skill routing. The trained task strings contain these tokens.
SKILL_KEYWORDS = {
    "bottle": ("bottle",),
    "bear": ("bear", "teddy", "stuffed"),
    "duck": ("duck",),
    "hat": ("hat", "cap"),
}


def route_skill(prompt: str, available: list[str], default: str) -> str:
    p = (prompt or "").lower()
    for skill in available:
        for kw in SKILL_KEYWORDS.get(skill, (skill,)):
            if kw in p:
                return skill
    return default


def _is_lora_name(name: str) -> bool:
    n = name.lower()
    return n.endswith("lora_a") or n.endswith("lora_b") or "lora" in n.split("/")[-2:][0:1]


# ── adapter (lora leaf) loading ─────────────────────────────────────────────


def load_adapter_leaves(ckpt_dir: str) -> dict:
    """Restore ONLY the lora_a/lora_b leaves from a checkpoint's params/, as a
    flat ``{"a/b/c/lora_a": np.ndarray}`` dict (paths joined with '/')."""
    import os

    import flax.traverse_util as ftu
    import numpy as np
    from openpi.models import model as _model
    from openpi.shared import download

    params_path = ckpt_dir.rstrip("/")
    if not params_path.endswith("params"):
        params_path = os.path.join(params_path, "params")
    params = _model.restore_params(download.maybe_download(params_path), restore_type=np.ndarray)
    flat = ftu.flatten_dict(params, sep="/")
    leaves = {k: np.asarray(v) for k, v in flat.items()
              if k.lower().endswith("lora_a") or k.lower().endswith("lora_b")}
    if not leaves:
        raise SystemExit(f"no lora_a/lora_b leaves under {params_path} — is this a *_lora ckpt?")
    n = sum(v.size for v in leaves.values())
    log.info("loaded %d adapter leaves (%.1fM params) from %s", len(leaves), n / 1e6, ckpt_dir)
    return leaves


def _suffix(path_tuple, k: int = 4) -> str:
    """Last k path components, lowercased and '/'-joined — used to match an nnx
    state path against a restored checkpoint key regardless of prefix wrapping."""
    parts = [str(c).lower() for c in path_tuple]
    return "/".join(parts[-k:])


class HotSwapper:
    """Mutates the model's nnx lora leaves on skill change and rebuilds the
    jitted sampler. Degrades to base-only (armed=False) on any introspection
    mismatch so the serve is never left dead."""

    def __init__(self, policy, adapters: dict[str, dict]):
        self.policy = policy
        self.adapters = adapters
        self.active: str | None = None
        self.armed = False
        self.model = getattr(policy, "_model", None)
        if self.model is None:
            log.warning("HOT-SWAP DISABLED — policy has no nnx _model (got attrs: %s)",
                        [a for a in vars(policy)][:12])
            return
        try:
            import jax
            from flax import nnx

            self._jax = jax
            self._nnx = nnx
            # Flatten the model's nnx state to {path_tuple: VariableState}.
            state = nnx.state(self.model)
            self._flat = dict(state.flat_state())
            lora_paths = [p for p, v in self._flat.items()
                          if str(p[-1]).lower() in ("lora_a", "lora_b")]
            if not lora_paths:
                log.warning("HOT-SWAP DISABLED — no lora_a/lora_b leaves in nnx state "
                            "(%d leaves total)", len(self._flat))
                return
            # Map each adapter's restored leaves onto the nnx state paths by
            # matching path suffixes (robust to module-wrapper prefixes).
            self._skill_states: dict[str, dict] = {}
            for skill, leaves in adapters.items():
                by_suffix = {}
                for k, arr in leaves.items():
                    by_suffix["/".join(k.lower().split("/")[-4:])] = arr
                mapped, missed = {}, 0
                for p in lora_paths:
                    arr = by_suffix.get(_suffix(p))
                    if arr is None:
                        missed += 1
                        continue
                    mapped[p] = jax.device_put(arr)
                if missed:
                    log.warning("skill %s: %d/%d lora leaves unmatched", skill, missed, len(lora_paths))
                self._skill_states[skill] = mapped
            self.armed = any(self._skill_states.values())
            if self.armed:
                # Synthetic "base" target: zero LoRA leaves => no adapter
                # contribution => the bare FFT-56k base (generalist) for objects
                # no trained skill fits.
                try:
                    self._skill_states[BASE_SKILL] = {
                        p: jax.device_put(jax.numpy.zeros_like(self._flat[p].value))
                        for p in lora_paths
                    }
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not build base zero-adapter state: %s", exc)
                log.info(">>> hot-swap armed · %d adapted leaves · skills=%s (+base)",
                         len(lora_paths), list(adapters))
            else:
                log.warning("HOT-SWAP DISABLED — no adapter leaves mapped onto nnx state")
        except Exception as exc:  # noqa: BLE001
            log.warning("HOT-SWAP DISABLED — nnx introspection failed: %s", exc)
            self.armed = False

    def _rebuild_sampler(self) -> None:
        """Rebuild policy._sample_actions so it samples with the updated state,
        the same way openpi builds it (module_jit), with plain-jit fallbacks."""
        nnx = self._nnx
        model = self.model
        try:
            from openpi.shared import nnx_utils
            self.policy._sample_actions = nnx_utils.module_jit(model.sample_actions)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("module_jit rebuild failed (%s); trying nnx.jit", exc)
        try:
            self.policy._sample_actions = nnx.jit(model.sample_actions)
        except Exception as exc:  # noqa: BLE001
            log.warning("nnx.jit rebuild failed (%s); using un-jitted sample_actions", exc)
            self.policy._sample_actions = model.sample_actions

    def ensure(self, skill: str) -> None:
        if not self.armed or skill == self.active or skill not in self._skill_states:
            return
        arrays = self._skill_states[skill]
        if not arrays:
            return
        nnx = self._nnx
        t0 = time.monotonic()
        try:
            for path, arr in arrays.items():
                self._flat[path].value = arr            # mutate the lora leaves
            new_state = nnx.State.from_flat_state(self._flat)
            nnx.update(self.model, new_state)           # push back into the module
            self._rebuild_sampler()                     # next infer uses new adapter
            self._jax.block_until_ready(list(arrays.values()))
        except Exception as exc:  # noqa: BLE001
            log.warning("swap to %s failed (%s) — keeping %s", skill, exc, self.active)
            return
        dt = (time.monotonic() - t0) * 1000.0
        log.info("swap skill %s -> %s  (%d leaves, %.0f ms)", self.active, skill, len(arrays), dt)
        self.active = skill


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="registered openpi config (pi05_fftlora)")
    ap.add_argument("--policy-dir", required=True, help="a full task ckpt dir to build the policy from")
    ap.add_argument("--adapter", action="append", default=[], metavar="skill=ckpt_dir",
                    help="repeatable: a skill name and the ckpt dir to pull its lora leaves from")
    ap.add_argument("--default-skill", default="bottle")
    ap.add_argument("--default-prompt", default="Picking up the bottle and placing it on the box")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    adapters_spec: dict[str, str] = {}
    for a in args.adapter:
        if "=" not in a:
            sys.exit(f"--adapter must be skill=ckpt_dir, got {a!r}")
        k, v = a.split("=", 1)
        adapters_spec[k.strip()] = v.strip()
    if not adapters_spec:
        sys.exit("at least one --adapter skill=ckpt_dir is required")

    # Heavy imports here so --help stays instant.
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    log.info(">>> building base policy from %s (config=%s)", args.policy_dir, args.config)
    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)

    log.info(">>> preloading %d adapters: %s", len(adapters_spec), list(adapters_spec))
    adapters = {skill: load_adapter_leaves(d) for skill, d in adapters_spec.items()}
    swapper = HotSwapper(policy, adapters)
    if not swapper.armed:
        log.error("FATAL: LoRA hot-swap could not arm. Refusing to serve a degraded "
                  "base-only policy (no per-object skill swapping). See the "
                  "HOT-SWAP DISABLED warning above for the cause.")
        sys.exit(2)
    default_skill = args.default_skill if args.default_skill in adapters else next(iter(adapters))
    swapper.ensure(default_skill)

    # Wrap infer so the adapter follows the prompt. openpi passes the observation
    # dict (with a "prompt" key) straight through; we route it to a skill and
    # swap before delegating to the real inference. An empty prompt falls back to
    # --default-prompt (so the documented default actually seeds routing).
    orig_infer = policy.infer
    available = list(adapters.keys())

    def infer(obs, **kw):
        prompt = ""
        if isinstance(obs, dict):
            prompt = obs.get("prompt") or obs.get("task") or ""
        if not prompt:
            prompt = args.default_prompt
        # A real prompt that matches no trained skill keyword routes to BASE_SKILL
        # (zero adapter) — the generalist FFT-56k base handles the novel object.
        swapper.ensure(route_skill(prompt, available, BASE_SKILL))
        return orig_infer(obs, **kw)

    policy.infer = infer  # type: ignore[method-assign]

    log.info(">>> serve_policy.py on :%d (fftlora hot-swap, skills=%s, default=%s)",
             args.port, available, default_skill)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port,
    )
    # Marker AFTER the server object exists (socket about to accept). openpi's
    # own "Creating server (host=...)" log also matches cluster.py's BOUND regex;
    # this is a belt-and-suspenders line emitted at the right point, not before.
    log.info(">>> server listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
