"""Risk-adjusted performance ratios.

All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame``.
NaN values are dropped before computation.
"""

from __future__ import annotations

import polars as pl

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series
from ruin.drawdown import max_drawdown
from ruin.returns import annualize_return
from ruin.volatility import annualize_volatility, downside_deviation


def sharpe_ratio(
    returns: ReturnInput,
    *,
    risk_free: float = 0.0,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    risk_free:
        Per-period risk-free rate. Default 0.0.
    periods_per_year:
        Number of periods in a year (e.g. 252 for daily).
    ddof:
        Delta degrees of freedom for volatility. Default 1.

    Returns
    -------
    float
        Sharpe ratio = annualized_excess_return / annualized_volatility.

    Notes
    -----
    ``risk_free`` is a *per-period* rate, not annualized. Use
    ``ruin.periods.annual_to_periodic`` to convert if needed.
    """
    r = to_series(returns)
    require_minimum_length(r, ddof + 1, "sharpe_ratio")
    excess = r - risk_free
    ann_excess = float(excess.mean()) * periods_per_year  # type: ignore[operator]
    ann_vol = float(excess.std(ddof=ddof)) * (periods_per_year**0.5)  # type: ignore[arg-type]
    if ann_vol == 0.0:
        return float("nan")
    return ann_excess / ann_vol


def sortino_ratio(
    returns: ReturnInput,
    *,
    risk_free: float = 0.0,
    threshold: float | None = None,
    periods_per_year: float,
) -> float:
    """Annualized Sortino ratio.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    risk_free:
        Per-period risk-free rate. Default 0.0.
    threshold:
        Minimum acceptable return for downside deviation. Defaults to ``risk_free``.
    periods_per_year:
        Number of periods in a year.

    Returns
    -------
    float
        Sortino ratio = annualized_excess_return / annualized_downside_deviation.

    Notes
    -----
    Downside deviation uses ``ddof=0`` (population convention). The denominator
    counts all periods in the denominator, not just downside periods.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "sortino_ratio")
    mar = risk_free if threshold is None else threshold
    excess_mean = float((r - risk_free).mean()) * periods_per_year  # type: ignore[operator]
    dd = downside_deviation(r, threshold=mar, ddof=0)
    ann_dd = dd * (periods_per_year**0.5)
    if ann_dd == 0.0:
        return float("nan")
    return excess_mean / ann_dd


def calmar_ratio(returns: ReturnInput, *, periods_per_year: float) -> float:
    """Calmar ratio: CAGR divided by absolute max drawdown.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    periods_per_year:
        Number of periods in a year.

    Returns
    -------
    float
        Calmar ratio. ``float('nan')`` if max drawdown is zero.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "calmar_ratio")
    ann_ret = annualize_return(r, periods_per_year=periods_per_year)
    mdd = max_drawdown(r)
    if mdd == 0.0:
        return float("nan")
    return ann_ret / abs(mdd)


def information_ratio(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Annualized information ratio.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.
    periods_per_year:
        Number of periods in a year.
    ddof:
        Delta degrees of freedom for tracking error. Default 1.

    Returns
    -------
    float
        IR = annualized_active_return / annualized_tracking_error.
    """
    from ruin._internal.validate import align_benchmark

    r, b = align_benchmark(returns, benchmark)
    active = r - b
    ann_active = float(active.mean()) * periods_per_year  # type: ignore[operator]
    ann_te = float(active.std(ddof=ddof)) * (periods_per_year**0.5)  # type: ignore[arg-type]
    if ann_te == 0.0:
        return float("nan")
    return ann_active / ann_te


def treynor_ratio(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    risk_free: float = 0.0,
    periods_per_year: float,
) -> float:
    """Annualized Treynor ratio.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.
    risk_free:
        Per-period risk-free rate. Default 0.0.
    periods_per_year:
        Number of periods in a year.

    Returns
    -------
    float
        Treynor ratio = annualized_excess_return / beta.
    """
    from ruin._internal.validate import align_benchmark
    from ruin.market import beta as compute_beta

    r, b = align_benchmark(returns, benchmark)
    ann_excess = float((r - risk_free).mean()) * periods_per_year  # type: ignore[operator]
    beta_val = compute_beta(r, b)
    if beta_val == 0.0:
        return float("nan")
    return ann_excess / beta_val


def omega_ratio(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains over losses above a threshold.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    float
        Omega ratio = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0)).
        ``float('nan')`` if there are no returns below the threshold.

    Notes
    -----
    Omega > 1 iff mean(r) > threshold (for any distribution with finite mean).
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "omega_ratio")
    gains = (r - threshold).clip(lower_bound=0.0).sum()
    losses = (threshold - r).clip(lower_bound=0.0).sum()
    if losses == 0.0:
        return float("nan")
    return float(gains) / float(losses)
