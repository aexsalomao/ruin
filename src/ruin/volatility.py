"""Volatility and dispersion measures. Inputs: pl.Series / np.ndarray / pl.DataFrame; NaNs dropped."""

from __future__ import annotations

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def volatility(returns: ReturnInput, *, ddof: int = 1) -> float:
    """Periodic standard deviation of returns (not annualized). `ddof=1` is sample std."""
    r = to_series(returns)
    require_minimum_length(r, ddof + 1, "volatility")
    return float(r.std(ddof=ddof))  # type: ignore[arg-type]


def annualize_volatility(
    returns: ReturnInput,
    *,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Annualized volatility via sqrt-of-time. Assumes i.i.d. returns; breaks under autocorrelation."""
    if periods_per_year <= 0:
        raise ValueError(f"'periods_per_year' must be positive; got {periods_per_year}")
    return volatility(returns, ddof=ddof) * (periods_per_year**0.5)


def downside_deviation(
    returns: ReturnInput,
    *,
    threshold: float = 0.0,
    ddof: int = 0,
) -> float:
    """Semi-std of returns below `threshold` (MAR). Denominator counts all periods (Sortino convention).

    Only `r < threshold` contributes to the sum of squared deviations, but all periods are in the
    denominator. Use `semi_deviation` if you want the denominator restricted to downside obs.
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
    """Std of negative returns only — both numerator and denominator restricted to `r < 0`."""
    r = to_series(returns)
    negative = r.filter(r < 0.0)
    if len(negative) == 0:
        return 0.0
    if len(negative) - ddof <= 0:
        raise ValueError(f"Not enough negative observations for ddof={ddof}; got {len(negative)}.")
    return float(negative.std(ddof=ddof))  # type: ignore[arg-type]
