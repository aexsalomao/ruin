"""Return computation functions.

All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame`` (one column
per strategy). NaN values are dropped before computation.

Sign convention: returns are dimensionless fractions (0.01 = 1%).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def from_prices(prices: ReturnInput, *, log: bool = False) -> pl.Series:
    """Compute simple or log returns from a price series.

    Parameters
    ----------
    prices:
        Price series (must be positive and finite).
    log:
        If ``True``, compute log returns: ln(P_t / P_{t-1}).
        If ``False`` (default), compute simple returns: P_t / P_{t-1} - 1.

    Returns
    -------
    pl.Series
        Return series of length ``len(prices) - 1``. NaNs dropped.

    Notes
    -----
    The first element is always dropped (no return for the initial price).
    """
    p = to_series(prices, name="prices")
    require_minimum_length(p, 2, "from_prices")
    if log:
        r = (p / p.shift(1)).log()
    else:
        r = p / p.shift(1) - 1.0
    return r.drop_nulls().drop_nans()


def total_return(returns: ReturnInput) -> float:
    """Compound all periodic returns into a single total return.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    float
        Total compounded return. E.g. 0.25 means +25%.

    Notes
    -----
    Computed as ``product(1 + r_i) - 1``.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "total_return")
    return float((1.0 + r).product()) - 1.0


def annualize_return(
    returns: ReturnInput,
    *,
    periods_per_year: float,
    method: str = "geometric",
) -> float:
    """Annualize a return series.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    periods_per_year:
        Number of periods in a year (e.g. 252 for daily, 12 for monthly).
    method:
        ``"geometric"`` (default): ``(1 + total_return)^(periods_per_year / n) - 1``.
        ``"arithmetic"``: ``mean(r) * periods_per_year``.

    Returns
    -------
    float
        Annualized return.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "annualize_return")
    if periods_per_year <= 0:
        raise ValueError(f"'periods_per_year' must be positive; got {periods_per_year}")
    if method == "geometric":
        n = len(r)
        tr = float((1.0 + r).product())
        return float(tr ** (periods_per_year / n)) - 1.0
    elif method == "arithmetic":
        return float(r.mean()) * periods_per_year  # type: ignore[operator]
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'geometric' or 'arithmetic'.")


def cagr(returns: ReturnInput, *, periods_per_year: float) -> float:
    """Compound annual growth rate — alias for geometric annualized return.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    periods_per_year:
        Number of periods in a year (e.g. 252 for daily, 12 for monthly).

    Returns
    -------
    float
        CAGR as a decimal fraction (e.g. 0.12 = 12% per year).
    """
    return annualize_return(returns, periods_per_year=periods_per_year, method="geometric")
