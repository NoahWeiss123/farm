"""Generalized & Specialized Experts (GSE) adapters for openpi / π0.5.

Port of **VLA-GSE** (Jiang et al., arXiv:2605.06175, "Boosting Parameter-
Efficient Fine-Tuning in VLA with Generalized and Specialized Experts") into
openpi's Flax-linen Gemma stack. Drop-in replacement for the LoRA ``Einsum`` /
``FeedForward`` in ``openpi/models/lora.py`` — same call signatures, same
batched factor shapes, so it composes with Gemma's existing einsum sites.

``patch_openpi_gse.py`` installs this file as ``openpi/models/gse.py`` and wires
the ``gemma_2b_gse`` / ``gemma_300m_gse`` variants.

────────────────────────────────────────────────────────────────────────────
Method (paper §3, adapted to π0.5)
────────────────────────────────────────────────────────────────────────────
A frozen pretrained weight W₀ = U Σ Vᵀ is split by its singular spectrum:

  • **Generalized expert** (always-on): the leading ``generalized_rank``
    singular components — the dominant, most-transferable subspace. A trainable
    low-rank adapter, but SVD-*initialized* so it starts as an exact
    reconstruction of that subspace and only gently adapts it (vs. LoRA's
    random near-zero init, which discards the prior).
  • **Specialized experts** (``num_specialized`` of them): the next
    ``num_specialized · expert_rank`` components, partitioned into contiguous
    rank-``expert_rank`` blocks — the residual subspace where task-specific
    control adaptation happens.
  • **Frozen backbone**: W₀ minus those reconstructions (Eq. 12 "backbone
    adjustment") → only the low-energy spectral tail stays frozen. Because the
    adapters reconstruct the dominant components at init, the forward at init
    ≈ W₀·x (behavior-preserving), then training adapts the dominant subspace
    without random-init shock (cf. LoRA) and without touching the whole 3.3B
    backbone (cf. full FT, which overfits 2-task data and forgets).

Trainable: the expert adapters (this module) + the action expert (a separate
gemma_300m, full fine-tuned). Frozen: the VLM backbone tail.

SVD initialization + backbone adjustment run in ``GSESVDWeightLoader``
(installed by the patch) because W₀ is only known after the checkpoint loads —
not at module ``setup``. Until then the factors hold a zero init for the ``b``
side, so the adapters contribute nothing and the model == the base checkpoint.

ROUTING. The paper routes tokens through a top-k subset of specialists. By
default this module runs the experts **dense** (all summed, equal weight),
which is correct-by-construction — mathematically identical to a single LoRA
adapter of rank ``generalized_rank + num_specialized·expert_rank`` — so it is
guaranteed to run and gets GSE's spectral-init benefit. Per-token routing is an
opt-in (``route=True``) experimental path: it is clean for the (…,D)-input
einsums (qkv/kv/q/ffn) but the attention-output einsum carries an extra head
axis, so validate routing on the cluster before relying on it.

WHERE SVD-INIT APPLIES. The two-step low-rank exactly reconstructs a weight only
when the matmul axes are the trailing two and no *leading* axis is contracted.
That holds for the attention input projections (qkv/q/kv) and the FFN dots
(gating + linear) — numerically verified to ~1e-14 — so SVD-init covers the bulk
of the VLM: the GSE adapters here cover q/k/v, and ``GSESVDWeightLoader`` also
PiSSA-initializes the FFN's plain LoRA adapters (see ``svd_init_pissa``). It does
NOT hold for the attention-output projection ``BTNH,NHD->BTD`` (it contracts the
leading head axis N), so the loader leaves that adapter at its zero init (a plain
trainable-from-zero LoRA-style delta, backbone unadjusted) — a no-op at init,
fully preserving W₀. Net: the dominant subspace is spectrally preserved across
almost the whole backbone, and only the output projection adapts from zero.
"""

import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp
import openpi.shared.array_typing as at


@struct.dataclass
class GSEConfig:
    """Configuration for a GSE adapter block."""

    # Rank of the always-on generalized expert (leading singular components).
    generalized_rank: int = 2
    # Number of specialized experts (residual singular blocks).
    num_specialized: int = 7
    # Rank of each specialized expert (paper's d). Total adapted rank r =
    # generalized_rank + num_specialized·expert_rank (paper: 2 + 7·2 = 16).
    expert_rank: int = 2
    # Per-token top-k routing over specialists (only used when route=True).
    top_k: int = 2
    # Enable per-token routing. Default dense (correct-by-construction; see
    # the module docstring). Turn on only after the cluster smoke test.
    route: bool = False
    # Initializer for the ``a`` factors before the SVD loader overwrites them.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # Axes of the weight the adapter acts on (last two, like LoRA).
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"  # einsum label for the low-rank axis (must not clash)

    @property
    def total_rank(self) -> int:
        return self.generalized_rank + self.num_specialized * self.expert_rank


def _make_lora_eqns(eqn: str, axes: tuple[int, int], label: str) -> tuple[str, str]:
    """Split ``eqn`` into the two low-rank einsums — identical to
    ``lora.Einsum._make_lora_eqns`` so the low-rank path matches LoRA exactly."""
    if label in eqn:
        raise ValueError(f"label {label!r} already in eqn: {eqn}")
    if not (m := re.match("(.*),(.*)->(.*)", eqn)):
        raise ValueError(f"Unsupported einsum eqn: {eqn}")
    lhs, rhs, out = m.groups()
    a_label, b_label = (rhs[x] for x in axes)
    a_rhs = rhs.replace(b_label, label)
    a_out = out.replace(b_label, label)
    eqn_a = f"{lhs},{a_rhs}->{a_out}"
    b_rhs = rhs.replace(a_label, label)
    eqn_b = f"{a_out},{b_rhs}->{out}"
    return eqn_a, eqn_b


def _factor_shapes(shape: tuple[int, ...], axes: tuple[int, int], rank: int) -> tuple[list[int], list[int]]:
    """LoRA factor shapes: full weight shape with one axis replaced by ``rank``
    (a: axes[1]→rank, b: axes[0]→rank). Matches lora.Einsum exactly."""
    sa, sb = list(shape), list(shape)
    sa[axes[1]] = rank
    sb[axes[0]] = rank
    return sa, sb


class Einsum(nn.Module):
    """Einsum with GSE adapters. Drop-in for the Gemma / LoRA ``Einsum``."""

    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    gse_config: GSEConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)
        if (cfg := self.gse_config) is not None:
            ga, gb = _factor_shapes(self.shape, cfg.axes, cfg.generalized_rank)
            self.gen_a = self.param("gse_gen_a", cfg.init_fn, tuple(ga))
            self.gen_b = self.param("gse_gen_b", nn.initializers.zeros, tuple(gb))
            sa, sb = _factor_shapes(self.shape, cfg.axes, cfg.expert_rank)
            self.spec_a = self.param("gse_spec_a", cfg.init_fn, (cfg.num_specialized, *sa))
            self.spec_b = self.param("gse_spec_b", nn.initializers.zeros, (cfg.num_specialized, *sb))
            if cfg.route:
                # Router over specialists; contracts the feature axis (axes[0]).
                router_shape = (self.shape[cfg.axes[0]], cfg.num_specialized)
                self.router = self.param("gse_router", nn.initializers.zeros, router_shape)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype
        result = jnp.einsum(eqn, x, self.w.astype(dtype))
        if (cfg := self.gse_config) is None:
            return result
        eqn_a, eqn_b = _make_lora_eqns(eqn, cfg.axes, cfg.label)

        # Generalized expert (always on): (x A_g) B_g.
        g = jnp.einsum(eqn_b, jnp.einsum(eqn_a, x, self.gen_a.astype(dtype)), self.gen_b.astype(dtype))
        result = result + g

        # Specialized experts. ``vmap`` the low-rank path over the expert axis,
        # then dense-sum (default) or route (experimental).
        def expert(a, b):
            return jnp.einsum(eqn_b, jnp.einsum(eqn_a, x, a.astype(dtype)), b.astype(dtype))

        spec = jax.vmap(expert)(self.spec_a, self.spec_b)  # (E, *out_shape)
        if not cfg.route:
            result = result + jnp.sum(spec, axis=0)
            return result

        # --- experimental per-token routing (route=True) ---
        logits = jnp.einsum("...d,de->...e", x, self.router.astype(dtype))  # (..., E)
        if cfg.top_k < cfg.num_specialized:
            kth = jnp.sort(logits, axis=-1)[..., -cfg.top_k][..., None]
            logits = jnp.where(logits >= kth, logits, -jnp.inf)
        gate = jax.nn.softmax(logits, axis=-1)  # (..., E) over the lhs batch dims
        # Broadcast gate[...,i] onto each expert's output by aligning labels.
        lhs, _, out = re.match("(.*),(.*)->(.*)", eqn).groups()
        tok = [c for c in lhs if c in out]  # batch dims surviving to the output
        bshape = [result.shape[out.index(c)] if c in tok else 1 for c in out]
        gate_t = jnp.moveaxis(gate, -1, 0)  # (E, ...tok)
        gate_t = gate_t.reshape((cfg.num_specialized, *bshape))
        result = result + jnp.sum(spec * gate_t.astype(dtype), axis=0)
        return result


class FeedForward(nn.Module):
    """Gemma FeedForward with GSE adapters (dense, SVD-initialized). Drop-in for
    the LoRA ``FeedForward``. Routing is not applied to the FFN (the paper's
    experts are concentrated in attention).

    NOTE: optional / not used by the ``*_gse`` variants — those keep the FFN on
    the plain ``lora.FeedForward`` and let ``GSESVDWeightLoader`` PiSSA-init its
    adapters (``svd_init_pissa``), which gives the same SVD-init benefit through
    the proven LoRA path. This class exists for experiments that want the FFN to
    carry the explicit generalized/specialized split too."""

    features: int
    hidden_dim: int
    gse_config: GSEConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear", nn.initializers.lecun_normal(in_axis=-2, out_axis=-1), (self.hidden_dim, self.features)
        )
        if (cfg := self.gse_config) is not None:
            r = cfg.total_rank
            self.gate_a = self.param("gse_gate_a", cfg.init_fn, (2, self.features, r))
            self.gate_b = self.param("gse_gate_b", nn.initializers.zeros, (2, r, self.hidden_dim))
            self.lin_a = self.param("gse_lin_a", cfg.init_fn, (self.hidden_dim, r))
            self.lin_b = self.param("gse_lin_b", nn.initializers.zeros, (r, self.features))

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        ff_gate = self._dot(x, self.w_gating[0], 0)
        ff1 = self._dot(x, self.w_gating[1], 1)
        activations = nn.gelu(ff_gate) * ff1
        outputs = self._dot_linear(activations, self.w_linear)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, gate_idx: int) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if self.gse_config is None:
            return base
        return base + jnp.dot(jnp.dot(x, self.gate_a[gate_idx].astype(x.dtype)), self.gate_b[gate_idx].astype(x.dtype))

    def _dot_linear(self, x: at.Array, w: at.Array) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if self.gse_config is None:
            return base
        return base + jnp.dot(jnp.dot(x, self.lin_a.astype(x.dtype)), self.lin_b.astype(x.dtype))


def svd_init_factors(
    w0: jax.Array, axes: tuple[int, int], generalized_rank: int, num_specialized: int, expert_rank: int
):
    """Batched SVD initialization for one (possibly multi-dim) weight ``w0``.

    Treats ``w0`` as a stack of (a_dim, b_dim) matrices over its leading dims
    (the last two axes are the matmul axes, matching LoRA's ``axes=(-2,-1)``).
    Returns ``(gen_a, gen_b, spec_a, spec_b, w_adj)`` with the LoRA factor
    shapes, where ``B@A`` reconstructs each singular block exactly and ``w_adj``
    is the spectral tail (Eq. 12) so the dense forward at init reconstructs W₀.
    """
    if axes != (-2, -1):
        raise ValueError(f"svd_init_factors supports axes=(-2,-1), got {axes}")
    lead = w0.shape[:-2]
    a_dim, b_dim = w0.shape[-2], w0.shape[-1]
    flat = w0.reshape((-1, a_dim, b_dim))            # (L, a, b)
    u, s, vt = jnp.linalg.svd(flat, full_matrices=False)  # u (L,a,k) s (L,k) vt (L,k,b)
    sq = jnp.sqrt(s)

    def block(lo, hi):
        # a = U[:,:,lo:hi]·√Σ  → (L, a, r);  b = √Σ·V[lo:hi] → (L, r, b)
        a = u[:, :, lo:hi] * sq[:, None, lo:hi]
        b = vt[:, lo:hi, :] * sq[:, lo:hi, None]
        return a, b

    rg, d, e = generalized_rank, expert_rank, num_specialized
    gen_a, gen_b = block(0, rg)
    recon = jnp.einsum("lar,lrb->lab", gen_a, gen_b)
    spec_a_list, spec_b_list = [], []
    for i in range(e):
        lo = rg + i * d
        a, b = block(lo, lo + d)
        spec_a_list.append(a)
        spec_b_list.append(b)
        recon = recon + jnp.einsum("lar,lrb->lab", a, b)
    w_adj = (flat - recon).reshape(w0.shape)

    gen_a = gen_a.reshape((*lead, a_dim, rg))
    gen_b = gen_b.reshape((*lead, rg, b_dim))
    spec_a = jnp.stack([a.reshape((*lead, a_dim, d)) for a in spec_a_list])  # (E, *lead, a, d)
    spec_b = jnp.stack([b.reshape((*lead, d, b_dim)) for b in spec_b_list])  # (E, *lead, d, b)
    return gen_a, gen_b, spec_a, spec_b, w_adj


def svd_init_pissa(w0: jax.Array, rank: int):
    """PiSSA (SVD) initialization for a single LoRA adapter — used to extend the
    GSE spectral-init mechanism (its dominant lever, paper ablation +13 pts) to
    the FFN, where the gemma FeedForward uses a plain LoRA adapter.

    Treats the last two axes of ``w0`` as the matmul axes (batched over leading
    dims). Returns ``(lora_a, lora_b, w_adj)`` such that ``lora_a @ lora_b``
    reconstructs the top-``rank`` singular subspace and ``w_adj = w0 - that``
    (Eq. 12). Assumes the LoRA scaling is 1 (alpha == rank), so at init the
    forward ``w_adj·x + (x·a)·b`` exactly equals ``w0·x``.
    """
    lead = w0.shape[:-2]
    a_dim, b_dim = w0.shape[-2], w0.shape[-1]
    flat = w0.reshape((-1, a_dim, b_dim))
    u, s, vt = jnp.linalg.svd(flat, full_matrices=False)
    sq = jnp.sqrt(s[:, :rank])
    a = u[:, :, :rank] * sq[:, None, :]      # (L, a, r)
    b = vt[:, :rank, :] * sq[:, :, None]     # (L, r, b)
    w_adj = (flat - jnp.einsum("lar,lrb->lab", a, b)).reshape(w0.shape)
    return a.reshape((*lead, a_dim, rank)), b.reshape((*lead, rank, b_dim)), w_adj
