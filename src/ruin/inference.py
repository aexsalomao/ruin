"""Statistical inference for performance metrics — critical for small live samples. NaNs dropped."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import polars as pl

from ruin._internal.normal import norm_ppf
from ruin._internal.validate import ReturnInput, require_minimum_length, to_series
from ruin.distribution import autocorrelation
from ruin.ratios import sharpe_ratio


def sharpe_standard_error(
    returns: ReturnInput,
    *,
    periods_per_year: float,
) -> float:
    """Lo (2002) autocorrelation-adjusted SE of the annualized Sharpe ratio.

    Lo (2002), "The Statistics of Sharpe Ratios," Financial Analysts Journal:
    `SE(SR_ann) ≈ sqrt((1 + 2*rho_1 * SR_q^2) / T) * sqrt(q)`, where `q = periods_per_year`
    and `rho_1` is lag-1 autocorrelation. Reduces to `sqrt(q/T)` under i.i.d. returns.
    """
    r = to_series(returns)
    require_minimum_length(r, 4, "sharpe_standard_error")
    n = len(r)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
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
    """Asymptotic `(lower, upper)` CI for the annualized Sharpe ratio."""
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
    """Bootstrap CI for a scalar metric — returns `(point, lower, upper)`.

    Resamples `returns` with replacement and computes `fn(resample)` each time. `fn` must
    accept a pl.Series as first argument. `point` is `fn(returns)` on the original data.
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
        except (ValueError, ZeroDivisionError):
            continue
    if not estimates:
        return (point, float("nan"), float("nan"))
    estimates.sort()
    alpha = 1.0 - confidence
    lo_idx = int(math.floor(alpha / 2.0 * len(estimates)))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * len(estimates))) - 1
    hi_idx = min(hi_idx, len(estimates) - 1)
    return (point, estimates[lo_idx], estimates[hi_idx])
