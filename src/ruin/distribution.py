"""Return distribution shape metrics. NaNs dropped."""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series

__all__ = [
    "JarqueBeraResult",
    "autocorrelation",
    "excess_kurtosis",
    "jarque_bera",
    "skewness",
]


@dataclass(frozen=True)
class JarqueBeraResult:
    """Jarque-Bera result. `statistic` is the JB value; `p_value` is the asymptotic chi2(2) p-value."""

    statistic: float
    p_value: float


def skewness(returns: ReturnInput, *, bias: bool = False) -> float:
    """Skewness (third standardized moment). Negative = left-skewed; zero = symmetric.

    `bias=False` (default): Fisher-Pearson unbiased (SAS/SPSS convention).
    `bias=True`: biased population estimator `mean((r-mean)^3) / std^3`.
    """
    r = to_series(returns)
    require_minimum_length(r, 3, "skewness")
    n = len(r)
    mu = float(r.mean())  # type: ignore[arg-type]
    sigma = float(r.std(ddof=1))  # type: ignore[arg-type]
    # Guard against floating-point noise around zero variance: unique() on a
    # truly constant series is exactly one element.
    if sigma == 0.0 or r.n_unique() <= 1:
        return float("nan")
    cube = ((r - mu) / sigma) ** 3
    m3 = float(cube.mean())  # type: ignore[arg-type]
    if bias:
        return m3
    # Fisher-Pearson unbiased correction (SAS/SPSS convention):
    # skew = n**2 / ((n-1)*(n-2)) * mean(((x - mean) / std)**3)
    return (n**2) / ((n - 1) * (n - 2)) * m3


def excess_kurtosis(returns: ReturnInput, *, bias: bool = False) -> float:
    """Excess (Fisher) kurtosis. Normal = 0; positive = fatter tails.

    `bias=False` (default): unbiased Fisher correction (Excel KURT convention).
    `bias=True`: biased `mean((r-mean)^4) / std^4 - 3`.
    """
    r = to_series(returns)
    require_minimum_length(r, 4, "excess_kurtosis")
    n = len(r)
    mu = float(r.mean())  # type: ignore[arg-type]
    sigma = float(r.std(ddof=1))  # type: ignore[arg-type]
    if sigma == 0.0 or r.n_unique() <= 1:
        return float("nan")
    quad = ((r - mu) / sigma) ** 4
    m4 = float(quad.mean())  # type: ignore[arg-type]
    if bias:
        return m4 - 3.0
    # Unbiased (Fisher) excess kurtosis (SAS/SPSS/Excel KURT convention):
    # (n*(n+1)) / ((n-1)*(n-2)*(n-3)) * sum(z**4) - 3*(n-1)**2 / ((n-2)*(n-3))
    sum_quad = float(quad.sum())
    return n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * sum_quad - 3.0 * (n - 1) ** 2 / (
        (n - 2) * (n - 3)
    )


def jarque_bera(returns: ReturnInput) -> JarqueBeraResult:
    """Jarque-Bera normality test. `JB = n/6 * (S^2 + K^2/4)`; asymptotic chi2(2) under the null.

    Uses the exact chi2(2) CDF (`1 - exp(-x/2)`) — no SciPy needed.
    """
    r = to_series(returns)
    require_minimum_length(r, 8, "jarque_bera")
    n = len(r)
    s = skewness(r, bias=True)
    k = excess_kurtosis(r, bias=True)
    jb_stat = n / 6.0 * (s**2 + k**2 / 4.0)

    # p-value: P(chi2(2) > jb_stat) using regularized upper incomplete gamma
    # chi2(2) CDF = 1 - exp(-x/2) — exact for dof=2
    p_val = math.exp(-jb_stat / 2.0) if jb_stat >= 0.0 else 1.0

    return JarqueBeraResult(statistic=jb_stat, p_value=p_val)


def autocorrelation(returns: ReturnInput, *, lag: int = 1) -> float:
    """Lag-`k` serial autocorrelation: `corr(r[t], r[t - lag])`.

    Positive autocorrelation may indicate stale pricing or momentum and invalidates sqrt-of-time
    annualization used by `ruin.volatility.annualize_volatility`.
    """
    r = to_series(returns)
    require_minimum_length(r, lag + 2, "autocorrelation")
    if lag < 1:
        raise ValueError(f"'lag' must be >= 1; got {lag}")
    original = r.slice(lag)
    lagged = r.slice(0, len(r) - lag)
    df = pl.DataFrame({"a": original, "b": lagged})
    return float(df.select(pl.corr("a", "b")).item())
