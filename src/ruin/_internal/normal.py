"""Gaussian helpers using stdlib statistics.NormalDist — no SciPy dependency."""

from __future__ import annotations

from statistics import NormalDist

_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def norm_ppf(p: float) -> float:
    """Percent point function (inverse CDF) of the standard normal distribution.

    Parameters
    ----------
    p:
        Probability in (0, 1).

    Returns
    -------
    float
        The z-score such that P(Z <= z) = p.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1); got {p}")
    return _STANDARD_NORMAL.inv_cdf(p)


def norm_cdf(x: float) -> float:
    """CDF of the standard normal distribution at *x*.

    Parameters
    ----------
    x:
        Real-valued input.

    Returns
    -------
    float
        P(Z <= x) for Z ~ N(0, 1).
    """
    return _STANDARD_NORMAL.cdf(x)


def norm_pdf(x: float) -> float:
    """PDF of the standard normal distribution at *x*.

    Parameters
    ----------
    x:
        Real-valued input.

    Returns
    -------
    float
        Density at x for Z ~ N(0, 1).
    """
    return _STANDARD_NORMAL.pdf(x)
