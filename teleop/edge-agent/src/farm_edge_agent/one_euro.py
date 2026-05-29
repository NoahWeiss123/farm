"""1€ filter — speed-adaptive low-pass for noisy interactive input.

Reference: Casiez, Roussel, Vogel, *1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems* (CHI 2012).
Canonical Python implementation:
https://github.com/jaantollander/OneEuroFilter

The filter is a low-pass on the signal whose cutoff frequency rises
with the (smoothed) derivative of the signal. When the input is slow
or stationary, ``min_cutoff`` dominates and the output is heavily
smoothed; when the input moves fast, ``beta * |dx|`` raises the
cutoff so the filter introduces less lag. This gives the user the
visual jitter rejection of a low-pass without the laggy feel a
single-cutoff low-pass would have during fast motion.

Two classes are exposed:

* :class:`OneEuroFilter` for a single scalar channel.
* :class:`OneEuroFilterND` for a vector — ``n`` independent channels
  that share a sample clock. Used for the 6-DoF joint stream into the
  xArm streaming backend.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exp_smooth(a: float, x: float, x_prev: float) -> float:
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    """Scalar 1€ filter.

    Parameters
    ----------
    min_cutoff:
        Cutoff (Hz) at zero speed. Lower → more smoothing at rest;
        also more lag during slow motion. Typical 0.5–2.0 for cursor
        input; lower for very noisy signals.
    beta:
        Speed coefficient. Larger → cutoff rises faster with motion,
        i.e. less lag when active. Typical 0.0–0.1.
    d_cutoff:
        Cutoff (Hz) for the derivative estimate itself. 1.0 is the
        canonical default.
    """

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float = 0.0

    def reset(self, *, x0: float | None = None, t0: float | None = None) -> None:
        """Drop history. If ``x0`` is given the next call is treated as a
        warm sample and returns ``x0`` unchanged; otherwise the first
        post-reset sample seeds the filter."""
        self._x_prev = x0
        self._dx_prev = 0.0
        self._t_prev = 0.0 if t0 is None else float(t0)

    def __call__(self, t: float, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = float(x)
            self._t_prev = float(t)
            return float(x)
        t_e = float(t) - self._t_prev
        if t_e <= 0.0:
            # Out-of-order or duplicate timestamp — hold last output.
            return self._x_prev

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (float(x) - self._x_prev) / t_e
        dx_hat = _exp_smooth(a_d, dx, self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exp_smooth(a, float(x), self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = float(t)
        return x_hat


class OneEuroFilterND:
    """N independent 1€ channels sharing one sample clock.

    Constructed once for a given dimensionality. Pass a length-``n``
    sequence to ``__call__``; get back a length-``n`` list. The 6-DoF
    joint vector from the Quest teleop bridge is the primary client.
    """

    def __init__(
        self,
        n: int,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        if n <= 0:
            raise ValueError("OneEuroFilterND needs at least one channel")
        self._filters = [
            OneEuroFilter(
                min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
            )
            for _ in range(n)
        ]

    @property
    def n(self) -> int:
        return len(self._filters)

    def reset(
        self,
        *,
        x0: Sequence[float] | None = None,
        t0: float | None = None,
    ) -> None:
        for i, f in enumerate(self._filters):
            f.reset(x0=(x0[i] if x0 is not None else None), t0=t0)

    def set_params(
        self,
        *,
        min_cutoff: float | None = None,
        beta: float | None = None,
        d_cutoff: float | None = None,
    ) -> None:
        """Hot-update the per-channel filter parameters in place.

        Leaves filter state (last sample, derivative, timestamp)
        untouched, so the new settings take effect on the next
        ``__call__`` without a discontinuity.
        """
        for f in self._filters:
            if min_cutoff is not None:
                f.min_cutoff = float(min_cutoff)
            if beta is not None:
                f.beta = float(beta)
            if d_cutoff is not None:
                f.d_cutoff = float(d_cutoff)

    def get_params(self) -> dict[str, float]:
        f = self._filters[0]
        return {
            "min_cutoff": f.min_cutoff,
            "beta": f.beta,
            "d_cutoff": f.d_cutoff,
        }

    def __call__(self, t: float, x: Sequence[float]) -> list[float]:
        if len(x) != len(self._filters):
            raise ValueError(
                f"OneEuroFilterND got len(x)={len(x)}, expected {len(self._filters)}"
            )
        return [f(t, xi) for f, xi in zip(self._filters, x, strict=True)]
