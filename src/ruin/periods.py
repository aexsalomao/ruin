"""Time-slicing helpers and rate conversion. Period slicers require a date column; rate fns are pure arithmetic."""

from __future__ import annotations

import datetime

import polars as pl


def mtd(
    returns: pl.DataFrame | pl.Series,
    *,
    date_col: str,
    as_of: datetime.date | None = None,
) -> pl.DataFrame | pl.Series:
    """Month-to-date slice. `as_of` defaults to today; `returns` must carry a date dtype / column."""
    ref = as_of or datetime.date.today()
    month_start = ref.replace(day=1)
    if isinstance(returns, pl.Series):
        series_mask = (returns >= month_start) & (returns <= ref)
        return returns.filter(series_mask)
    expr_mask = (pl.col(date_col) >= month_start) & (pl.col(date_col) <= ref)
    return returns.filter(expr_mask)


def qtd(
    returns: pl.DataFrame | pl.Series,
    *,
    date_col: str,
    as_of: datetime.date | None = None,
) -> pl.DataFrame | pl.Series:
    """Quarter-to-date slice. `as_of` defaults to today."""
    ref = as_of or datetime.date.today()
    quarter_start_month = ((ref.month - 1) // 3) * 3 + 1
    quarter_start = ref.replace(month=quarter_start_month, day=1)
    if isinstance(returns, pl.Series):
        series_mask = (returns >= quarter_start) & (returns <= ref)
        return returns.filter(series_mask)
    expr_mask = (pl.col(date_col) >= quarter_start) & (pl.col(date_col) <= ref)
    return returns.filter(expr_mask)


def ytd(
    returns: pl.DataFrame | pl.Series,
    *,
    date_col: str,
    as_of: datetime.date | None = None,
) -> pl.DataFrame | pl.Series:
    """Year-to-date slice. `as_of` defaults to today."""
    ref = as_of or datetime.date.today()
    year_start = ref.replace(month=1, day=1)
    if isinstance(returns, pl.Series):
        series_mask = (returns >= year_start) & (returns <= ref)
        return returns.filter(series_mask)
    expr_mask = (pl.col(date_col) >= year_start) & (pl.col(date_col) <= ref)
    return returns.filter(expr_mask)


def trailing(
    returns: pl.DataFrame | pl.Series,
    *,
    n: int,
    date_col: str | None = None,
) -> pl.DataFrame | pl.Series:
    """Last `n` rows. `date_col` is unused — included for API symmetry; pass a sorted DataFrame."""
    if n <= 0:
        raise ValueError(f"'n' must be a positive integer; got {n}")
    return returns.tail(n)


def since_inception(returns: pl.DataFrame | pl.Series) -> pl.DataFrame | pl.Series:
    """Identity function — included for API symmetry."""
    return returns


_FREQUENCY_PERIODS: dict[str, int] = {
    "D": 252,
    "W": 52,
    "M": 12,
    "Q": 4,
    "A": 1,
    "Y": 1,
}


def periods_per_year_for(frequency: str) -> int:
    """Conventional periods/year for a frequency. D=252, W=52, M=12, Q=4, A/Y=1. Raises on unknown."""
    key = frequency.upper()
    if key not in _FREQUENCY_PERIODS:
        raise ValueError(
            f"Unknown frequency '{frequency}'. Valid options: {list(_FREQUENCY_PERIODS.keys())}"
        )
    return _FREQUENCY_PERIODS[key]


def annual_to_periodic(rate: float, *, periods_per_year: float) -> float:
    """Geometric convert annual -> periodic: `(1 + rate)^(1/periods_per_year) - 1`."""
    if periods_per_year <= 0:
        raise ValueError(f"'periods_per_year' must be positive; got {periods_per_year}")
    return (1.0 + rate) ** (1.0 / periods_per_year) - 1.0


def periodic_to_annual(rate: float, *, periods_per_year: float) -> float:
    """Geometric convert periodic -> annual: `(1 + rate)^periods_per_year - 1`."""
    if periods_per_year <= 0:
        raise ValueError(f"'periods_per_year' must be positive; got {periods_per_year}")
    return (1.0 + rate) ** periods_per_year - 1.0
