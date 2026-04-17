"""Return computation. Inputs: Series/ndarray/DataFrame; NaNs dropped. Fractions (0.01 = 1%)."""

from __future__ import annotations

import polars as pl

from ruin._internal.validate import (
    FLOAT_DTYPE,
    ReturnInput,
    require_minimum_length,
    require_strictly_positive,
    to_series,
)


def from_prices(prices: ReturnInput, *, log: bool = False) -> pl.Series:
    """Simple (default) or log returns from a positive price series. Length is `len(prices) - 1`."""
    p = to_series(prices, name="prices")
    require_minimum_length(p, 2, "from_prices")
    if (p <= 0.0).any():
        raise ValueError("'prices' must be strictly positive for return computation.")
    r = (p / p.shift(1)).log() if log else p / p.shift(1) - 1.0
    return r.drop_nulls().drop_nans().cast(FLOAT_DTYPE).rename("returns")


def total_return(returns: ReturnInput) -> float:
    """Compounded total return: `product(1 + r) - 1`."""
    r = to_series(returns)
    require_minimum_length(r, 1, "total_return")
    return float((1.0 + r).product()) - 1.0


def annualize_return(
    returns: ReturnInput,
    *,
    periods_per_year: float,
    method: str = "geometric",
) -> float:
    """Annualize returns by "geometric" (default) or "arithmetic" method.

    Geometric: `(1 + total)^(periods_per_year / n) - 1`. Returns NaN if total return <= -1 (ruin).
    Arithmetic: `mean(r) * periods_per_year`.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "annualize_return")
    require_strictly_positive(periods_per_year, "periods_per_year")
    if method == "geometric":
        n = len(r)
        tr = float((1.0 + r).product())
        # Total return can be <= 0 (ruin); fractional powers of negative
        # values are undefined. Return NaN rather than raising.
        if tr <= 0.0:
            return float("nan")
        return float(tr ** (periods_per_year / n)) - 1.0
    if method == "arithmetic":
        return float(r.mean()) * periods_per_year  # type: ignore[operator]
    raise ValueError(f"Unknown method '{method}'; choose 'geometric' or 'arithmetic'.")


def cagr(returns: ReturnInput, *, periods_per_year: float) -> float:
    """Compound annual growth rate — alias for geometric `annualize_return`."""
    return annualize_return(returns, periods_per_year=periods_per_year, method="geometric")
