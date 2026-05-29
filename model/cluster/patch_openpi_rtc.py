"""Patch openpi for server-side Real-Time Chunking (RTC).

RTC ("Real-Time Execution of Action Chunking Flow Policies", Black et al.,
arXiv:2506.07339) makes consecutive action chunks join smoothly by guiding the
flow-matching denoiser of the *next* chunk to stay continuous with the *previous*
one over their overlap — "freezing" the actions that will execute during the
inference delay and softly "inpainting" the rest.

This script edits two files in the openpi checkout in place (idempotent — safe to
re-run; it detects an existing patch and only re-verifies):

  src/openpi/models/pi0.py
    * adds `get_prefix_weights(start, end, total, schedule)`  (soft mask, Eq. 5)
    * rewrites `Pi0.sample_actions` to accept RTC kwargs and run a prefix-guided
      denoising loop (jax.vjp through the clean-action estimate).

  src/openpi/policies/policy.py
    * `Policy` keeps the previous *raw* (normalized, model-space) chunk as state
    * `Policy.infer` reads RTC control fields off the obs dict (rtc_reset /
      rtc_offset / rtc_delay), aligns the stored chunk to the new time frame, and
      passes the guidance kwargs into `sample_actions`.

The guidance algorithm is ported from Physical Intelligence's reference
implementation (github.com/Physical-Intelligence/real-time-chunking-kinetix,
src/model.py `realtime_action`), adapted to pi0.py's reversed flow-time
convention (here t=1 is noise and t=0 is the target; the reference uses the
opposite). The port is convention-independent in the guidance: the clean-action
estimate x_1 and its Jacobian wrt x_t are the same function, and the paper's
guidance weight is symmetric under t -> 1 - t.

Backward-compatible: with no RTC fields on the obs dict, `sample_actions` runs the
original plain Euler loop and `infer` behaves exactly as before.

Usage (on the login pod, inside the openpi checkout's env is NOT required — this
only edits + py_compiles source):

    python patch_openpi_rtc.py --openpi-root /home/nhweiss/farm-train/openpi
"""
# ruff: noqa: E501  — the GET_PREFIX_WEIGHTS / NEW_SAMPLE_ACTIONS / NEW_INFER
# string constants embed openpi source verbatim; their comment lines must match
# upstream byte-for-byte, so we don't reflow them to the 120-col limit.

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

# ── pi0.py edits ──────────────────────────────────────────────────────────────

GET_PREFIX_WEIGHTS = '''\
def get_prefix_weights(
    start: at.Int[at.Array, ""] | int,
    end: at.Int[at.Array, ""] | int,
    total: int,
    schedule: str,
) -> jax.Array:
    """Real-Time Chunking soft guidance weights over a chunk of length `total`.

    Ported verbatim from Physical Intelligence's real-time-chunking-kinetix
    (src/model.py `get_prefix_weights`). `start` (= inference delay) is where the
    chunk begins to be allowed to change; `end` (= prefix attention horizon, i.e.
    the overlap with the previous chunk) is where it stops attending to the
    previous chunk. With start=2, end=6, total=10 and schedule "linear" the
    weights are [1, 1, 4/5, 3/5, 2/5, 1/5, 0, 0, 0, 0]. `end` takes precedence:
    if end < start, start is pushed down to end (so end=0 ignores the prefix).
    """
    start = jnp.minimum(start, end)
    arange = jnp.arange(total)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (arange < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        w = jnp.clip((start - 1 - arange) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(arange >= end, 0, w)


'''

NEW_SAMPLE_ACTIONS = '''    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        prev_action_chunk: at.Float[at.Array, "b ah ad"] | None = None,
        inference_delay: at.Int[at.Array, ""] | int = 0,
        prefix_attention_horizon: at.Int[at.Array, ""] | int = 0,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def velocity(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `full_attn_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask_s, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry
            if prev_action_chunk is None:
                # Plain flow-matching Euler step (original behaviour).
                return x_t + dt * velocity(x_t, time), time + dt
            # Real-Time Chunking: prefix-guided flow-matching denoising that
            # inpaints the overlap with `prev_action_chunk` so consecutive chunks
            # join smoothly. Ported from Physical Intelligence's reference
            # (real-time-chunking-kinetix, src/model.py `realtime_action`),
            # adapted to this file's reversed flow-time convention (here t=1 is
            # noise and t=0 is the target; the reference uses t=0 noise -> t=1
            # target). The guidance is convention-independent: the clean-action
            # estimate `x_1` and its Jacobian wrt `x_t` are the same function, and
            # the paper's guidance weight is symmetric under t -> 1 - t.
            def denoiser(x):
                v = velocity(x, time)
                # one-step clean-action estimate in this convention: x_1 = x - t * v
                return x - time * v, v

            x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            weights = get_prefix_weights(
                inference_delay, prefix_attention_horizon, self.action_horizon, prefix_attention_schedule
            )
            error = (prev_action_chunk - x_1) * weights[None, :, None]
            (correction,) = vjp_fun(error)
            # paper Eq. 4 guidance weight, written in this file's time convention:
            inv_r2 = ((1.0 - time) ** 2 + time**2) / (time**2)
            c = jnp.nan_to_num(time / (1.0 - time), posinf=max_guidance_weight)
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
            v_guided = v_t - guidance_weight * correction
            return x_t + dt * v_guided, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0'''

# ── policy.py edits ─────────────────────────────────────────────────────────

OLD_METADATA_LINE = "        self._metadata = metadata or {}\n"
NEW_METADATA_LINE = (
    "        self._metadata = metadata or {}\n"
    "        # RTC: previous raw (normalized, model-space) action chunk, kept so the\n"
    "        # next sample can be guided to stay continuous with it. Per-instance\n"
    "        # state -> assumes a single client per Policy (true for serve_policy).\n"
    "        self._rtc_prev = None\n"
)

NEW_INFER = '''    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # --- Real-Time Chunking (server-side state) ---------------------------
        # The client may attach RTC control fields to the obs dict. They are not
        # model inputs, so strip them before the input transform. RTC keeps the
        # previous *raw* (normalized, model-space) action chunk on the server and
        # guides the next sample to stay continuous with it over the overlap.
        #   rtc_offset : steps advanced along the trajectory since the obs that
        #                produced the stored previous chunk (= re-plan stride)
        #   rtc_delay  : inference delay in steps -> size of the hard-frozen prefix
        #   rtc_reset  : drop stored state (first chunk of an episode)
        rtc_reset = bool(inputs.pop("rtc_reset", False)) if isinstance(inputs, dict) else False
        rtc_offset = inputs.pop("rtc_offset", None) if isinstance(inputs, dict) else None
        rtc_delay = inputs.pop("rtc_delay", None) if isinstance(inputs, dict) else None
        if rtc_reset:
            self._rtc_prev = None
        # ----------------------------------------------------------------------

        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # --- RTC: turn the stored previous chunk into guidance kwargs ---------
        # Guidance runs in the model's raw (normalized, padded) action space, so
        # align the stored previous chunk to the new chunk's time frame by shifting
        # it `rtc_offset` steps; guide only the overlapping region. JAX-only.
        if not self._is_pytorch_model and rtc_offset is not None and self._rtc_prev is not None:
            prev = self._rtc_prev  # (1, H, ad), normalized model space
            horizon = int(prev.shape[1])
            offset = max(0, int(rtc_offset))
            overlap = max(0, horizon - offset)
            if overlap > 0:
                delay = max(0, min(int(rtc_delay) if rtc_delay is not None else 0, overlap))
                y = jnp.zeros_like(prev)
                y = y.at[:, :overlap, :].set(prev[:, offset : offset + overlap, :])
                sample_kwargs["prev_action_chunk"] = y
                sample_kwargs["inference_delay"] = jnp.asarray(delay, dtype=jnp.int32)
                sample_kwargs["prefix_attention_horizon"] = jnp.asarray(overlap, dtype=jnp.int32)
        # ----------------------------------------------------------------------

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        raw_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        # Remember this chunk (raw, normalized, batched) for the next RTC step.
        if not self._is_pytorch_model:
            self._rtc_prev = raw_actions
        outputs = {
            "state": inputs["state"],
            "actions": raw_actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs'''


def _replace_between(text: str, start_anchor: str, end_anchor: str, new: str, *, what: str) -> str:
    i = text.find(start_anchor)
    if i == -1:
        raise SystemExit(f"FATAL: start anchor for {what} not found:\n  {start_anchor!r}")
    j = text.find(end_anchor, i)
    if j == -1:
        raise SystemExit(f"FATAL: end anchor for {what} not found:\n  {end_anchor!r}")
    j_end = j + len(end_anchor)
    return text[:i] + new + text[j_end:]


def patch_pi0(path: Path) -> bool:
    src = path.read_text()
    if "def get_prefix_weights" in src:
        print(f"[rtc] pi0.py already patched: {path}")
        return False
    if "class Pi0(_model.BaseModel):" not in src:
        raise SystemExit("FATAL: could not find `class Pi0` in pi0.py")
    # 1) insert the soft-mask helper just before the Pi0 class.
    src = src.replace(
        "class Pi0(_model.BaseModel):",
        GET_PREFIX_WEIGHTS + "class Pi0(_model.BaseModel):",
        1,
    )
    # 2) rewrite sample_actions (anchored slice replace, robust to internals).
    src = _replace_between(
        src,
        "    @override\n    def sample_actions(",
        "        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))\n        return x_0",
        NEW_SAMPLE_ACTIONS,
        what="Pi0.sample_actions",
    )
    path.write_text(src)
    print(f"[rtc] patched pi0.py: {path}")
    return True


def patch_policy(path: Path) -> bool:
    src = path.read_text()
    if "_rtc_prev" in src:
        print(f"[rtc] policy.py already patched: {path}")
        return False
    if OLD_METADATA_LINE not in src:
        raise SystemExit("FATAL: could not find metadata init line in policy.py")
    src = src.replace(OLD_METADATA_LINE, NEW_METADATA_LINE, 1)
    src = _replace_between(
        src,
        "    @override\n    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]",
        "        return outputs",
        NEW_INFER,
        what="Policy.infer",
    )
    path.write_text(src)
    print(f"[rtc] patched policy.py: {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--openpi-root", type=Path, default=Path("/home/nhweiss/farm-train/openpi"))
    args = ap.parse_args()

    pi0 = args.openpi_root / "src/openpi/models/pi0.py"
    policy = args.openpi_root / "src/openpi/policies/policy.py"
    for p in (pi0, policy):
        if not p.is_file():
            raise SystemExit(f"FATAL: not found: {p}")

    patch_pi0(pi0)
    patch_policy(policy)

    # Verify both files still compile.
    for p in (pi0, policy):
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as exc:
            raise SystemExit(f"FATAL: {p} failed to compile after patch:\n{exc}") from exc
    print("[rtc] OK: pi0.py + policy.py patched and compile cleanly")

    # Quick structural sanity checks.
    pi0_src = pi0.read_text()
    pol_src = policy.read_text()
    checks = [
        ("get_prefix_weights helper", "def get_prefix_weights(" in pi0_src),
        ("sample_actions RTC kwargs", "prev_action_chunk:" in pi0_src),
        ("guided velocity term", "v_guided = v_t - guidance_weight * correction" in pi0_src),
        ("jax.vjp guidance", "jax.vjp(denoiser, x_t, has_aux=True)" in pi0_src),
        ("policy rtc state", "self._rtc_prev = None" in pol_src),
        ("policy rtc plumbing", 'inputs.pop("rtc_offset"' in pol_src),
        ("policy stores raw chunk", "self._rtc_prev = raw_actions" in pol_src),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'XX'}] {name}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
