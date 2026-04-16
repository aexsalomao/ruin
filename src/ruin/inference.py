"""Statistical inference for performance metrics.

Critical for live performance tracking where sample sizes are small.
All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import polars as pl

from ruin._internal.normal import norm_ppf
from ruin._internal.validate import ReturnInput, require_minimum_length, to_series
from ruin.distribution import autocorrelation


def sharpe_standard_error(
    returns: ReturnInput,
    *,
    periods_per_year: float,
) -> float:
    """Lo (2002) autocorrelation-adjusted standard error of the Sharpe ratio.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    periods_per_year:
        Number of periods in a year.

    Returns
    -------
    float
        Standard error of the annualized Sharpe ratio.

    Notes
    -----
    From Lo (2002): "The Statistics of Sharpe Ratios," Financial Analysts Journal.
    The formula accounts for first-order autocorrelation in returns:

        SE(SR_annual) ≈ sqrt((1 + 2*rho_1 * SR_q^2/q) / T) * sqrt(q)

    where q = periods_per_year and rho_1 is lag-1 autocorrelation.
    In the iid case this reduces to sqrt(1/T).
    """
    r = to_series(returns)
    require_minimum_length(r, 4, "sharpe_standard_error")
    n = len(r)
    mu = float(r.mean())  # type: ignore[arg-type]
    sigma = float(r.std(ddof=1))  # type: ignore[arg-type]
    if sigma == 0.0:
        return float("nan")
    sr_q = mu / sigma  # per-period Sharpe
    rho1 = autocorrelation(r, lag=1)
    # Lo (2002) eq (12): variance of annualized SR
    var_sr = (1.0 + (2.0 * rho1 * sr_q**2)) / n
    # Annualize: SE_annual = SE_q * sqrt(q)
    se_q = var_sr**0.5
    return se_q * (periods_per_year**0.5)


def sharpe_confidence_interval(
    returns: ReturnInput,
    *,
    periods_per_year: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Asymptotic confidence interval for the annualized Sharpe ratio.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    periods_per_year:
        Number of periods in a year.
    confidence:
        Confidence level. Default 0.95.

    Returns
    -------
    tuple[float, float]
        (lower, upper) confidence interval bounds.
    """
    from ruin.ratios import sharpe_ratio

    r = to_series(returns)
    sr = sharpe_ratio(r, periods_per_year=periods_per_year)
    se = sharpe_standard_error(r, periods_per_year=periods_per_year)
    z = norm_ppf((1.0 + confidence) / 2.0)
    return (sr - z * se, sr + z * se)


def bootstrap_metric(
    fn: Callable[..., float],
    returns: ReturnInput,
    *,
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Generic bootstrap for any scalar metric.

    Resamples with replacement from *returns* and computes *fn* on each resample.

    Parameters
    ----------
    fn:
        A scalar metric function with signature ``fn(returns, **kwargs) -> float``.
        Must accept a ``pl.Series`` as first argument.
    returns:
        Periodic return series. NaNs are dropped before bootstrapping.
    n_samples:
        Number of bootstrap replicates. Default 1000.
    confidence:
        Confidence level for the interval. Default 0.95.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        ``(point_estimate, lower, upper)`` where *point_estimate* is ``fn(returns)``
        evaluated on the original data, and *lower*/*upper* are the bootstrap CI bounds.
    """
    r = to_series(returns)
    require_minimum_length(r, 2, "bootstrap_metric")
    point = fn(r)
    rng = np.random.default_rng(seed)
    arr = r.to_numpy()
    n = len(arr)
    estimates: list[float] = []
    for _ in range(n_samples):
        sample = rng.choice(arr, size=n, replace=True)
        s = pl.Series("r", sample, dtype=pl.Float64)
        try:
            estimates.append(fn(s))
        except Exception:
            pass
    if not estimates:
        return (point, float("nan"), float("nan"))
    estimates.sort()
    alpha = 1.0 - confidence
    lo_idx = int(math.floor(alpha / 2.0 * len(estimates)))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * len(estimates))) - 1
    hi_idx = min(hi_idx, len(estimates) - 1)
    return (point, estimates[lo_idx], estimates[hi_idx])
