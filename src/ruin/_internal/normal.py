"""Gaussian helpers via stdlib `statistics.NormalDist` — no SciPy dependency."""

from __future__ import annotations

from statistics import NormalDist

_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF: z such that `P(Z <= z) = p` for `p` in (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1); got {p}")
    return _STANDARD_NORMAL.inv_cdf(p)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF: `P(Z <= x)` for `Z ~ N(0, 1)`."""
    return _STANDARD_NORMAL.cdf(x)


def norm_pdf(x: float) -> float:
    """Standard-normal PDF density at `x` for `Z ~ N(0, 1)`."""
    return _STANDARD_NORMAL.pdf(x)
