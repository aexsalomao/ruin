"""Volatility and dispersion measures.

All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame``.
NaN values are dropped before computation.
"""

from __future__ import annotations

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def volatility(returns: ReturnInput, *, ddof: int = 1) -> float:
    """Standard deviation of returns (periodic, not annualized).

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    ddof:
        Delta degrees of freedom. Default 1 (sample std).

    Returns
    -------
    float
        Periodic standard deviation.
    """
    r = to_series(returns)
    require_minimum_length(r, ddof + 1, "volatility")
    return float(r.std(ddof=ddof))  # type: ignore[arg-type]


def annualize_volatility(
    returns: ReturnInput,
    *,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Annualized volatility using the square-root-of-time rule.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    periods_per_year:
        Number of periods in a year (e.g. 252 for daily, 12 for monthly).
    ddof:
        Delta degrees of freedom for std. Default 1 (sample std).

    Returns
    -------
    float
        Annualized standard deviation.

    Notes
    -----
    Assumes i.i.d. returns. The sqrt-of-time rule breaks down when returns
    have significant autocorrelation; see ``ruin.distribution.autocorrelation``.
    """
    if periods_per_year <= 0:
        raise ValueError(f"'periods_per_year' must be positive; got {periods_per_year}")
    return volatility(returns, ddof=ddof) * (periods_per_year**0.5)


def downside_deviation(
    returns: ReturnInput,
    *,
    threshold: float = 0.0,
    ddof: int = 0,
) -> float:
    """Semi-standard deviation of returns below a threshold.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return (MAR). Default 0.0.
    ddof:
        Delta degrees of freedom. Default 0 (population std of downside returns).

    Returns
    -------
    float
        Downside deviation (periodic).

    Notes
    -----
    Only periods where ``r < threshold`` contribute. Periods at or above the
    threshold contribute zero to the sum of squared deviations but are included
    in the denominator (Sortino/Upside Potential convention). If you want the
    denominator to be only downside observations, use ``semi_deviation``.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "downside_deviation")
    n = len(r)
    if n - ddof <= 0:
        raise ValueError(f"Not enough observations for ddof={ddof}; got {n}.")
    downside = (r - threshold).clip(upper_bound=0.0)
    sum_sq = float((downside**2).sum())
    return (sum_sq / (n - ddof)) ** 0.5


def semi_deviation(returns: ReturnInput, *, ddof: int = 0) -> float:
    """Standard deviation of negative returns only.

    Unlike ``downside_deviation``, only periods with ``r < 0`` contribute to
    both numerator *and* denominator.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    ddof:
        Delta degrees of freedom. Default 0 (population std of negative returns).

    Returns
    -------
    float
        Standard deviation of the negative-return subset.
    """
    r = to_series(returns)
    negative = r.filter(r < 0.0)
    if len(negative) == 0:
        return 0.0
    if len(negative) - ddof <= 0:
        raise ValueError(f"Not enough negative observations for ddof={ddof}; got {len(negative)}.")
    return float(negative.std(ddof=ddof))  # type: ignore[arg-type]
