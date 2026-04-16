"""Time-slicing helpers and rate conversion utilities.

Period-slice functions require a Polars DataFrame/Series with a date column.
Rate conversion functions are pure arithmetic.
"""

from __future__ import annotations

import datetime
from typing import Union

import polars as pl


def mtd(
    returns: pl.DataFrame | pl.Series,
    *,
    date_col: str,
    as_of: datetime.date | None = None,
) -> pl.DataFrame | pl.Series:
    """Slice a DataFrame/Series to month-to-date returns.

    Parameters
    ----------
    returns:
        Polars DataFrame (must include *date_col*) or Series with a date dtype.
    date_col:
        Name of the date column.
    as_of:
        Reference date. Defaults to today.

    Returns
    -------
    pl.DataFrame | pl.Series
        Rows where date is within the same calendar month as *as_of*.
    """
    ref = as_of or datetime.date.today()
    month_start = ref.replace(day=1)
    if isinstance(returns, pl.Series):
        mask = (returns >= month_start) & (returns <= ref)
        return returns.filter(mask)
    mask = (pl.col(date_col) >= month_start) & (pl.col(date_col) <= ref)
    return returns.filter(mask)


def qtd(
    returns: pl.DataFrame | pl.Series,
    *,
    date_col: str,
    as_of: datetime.date | None = None,
) -> pl.DataFrame | pl.Series:
    """Slice a DataFrame/Series to quarter-to-date returns.

    Parameters
    ----------
    returns:
        Polars DataFrame (must include *date_col*) or Series with a date dtype.
    date_col:
        Name of the date column.
    as_of:
        Reference date. Defaults to today.

    Returns
    -------
    pl.DataFrame | pl.Series
        Rows within the current quarter up to *as_of*.
    """
    ref = as_of or datetime.date.today()
    quarter_start_month = ((ref.month - 1) // 3) * 3 + 1
    quarter_start = ref.replace(month=quarter_start_month, day=1)
    if isinstance(returns, pl.Series):
        mask = (returns >= quarter_start) & (returns <= ref)
        return returns.filter(mask)
    mask = (pl.col(date_col) >= quarter_start) & (pl.col(date_col) <= ref)
    return returns.filter(mask)


def ytd(
    returns: pl.DataFrame | pl.Series,
    *,
    date_col: str,
    as_of: datetime.date | None = None,
) -> pl.DataFrame | pl.Series:
    """Slice a DataFrame/Series to year-to-date returns.

    Parameters
    ----------
    returns:
        Polars DataFrame (must include *date_col*) or Series with a date dtype.
    date_col:
        Name of the date column.
    as_of:
        Reference date. Defaults to today.

    Returns
    -------
    pl.DataFrame | pl.Series
        Rows within the current calendar year up to *as_of*.
    """
    ref = as_of or datetime.date.today()
    year_start = ref.replace(month=1, day=1)
    if isinstance(returns, pl.Series):
        mask = (returns >= year_start) & (returns <= ref)
        return returns.filter(mask)
    mask = (pl.col(date_col) >= year_start) & (pl.col(date_col) <= ref)
    return returns.filter(mask)


def trailing(
    returns: pl.DataFrame | pl.Series,
    *,
    n: int,
    date_col: str | None = None,
) -> pl.DataFrame | pl.Series:
    """Return the last *n* rows of a DataFrame/Series.

    Parameters
    ----------
    returns:
        Polars DataFrame or Series.
    n:
        Number of trailing periods to include.
    date_col:
        Unused — included for API symmetry. Pass the sorted DataFrame directly.

    Returns
    -------
    pl.DataFrame | pl.Series
        Last *n* rows.
    """
    if n <= 0:
        raise ValueError(f"'n' must be a positive integer; got {n}")
    return returns.tail(n)


def since_inception(returns: pl.DataFrame | pl.Series) -> pl.DataFrame | pl.Series:
    """Return the full series — identity function included for API symmetry.

    Parameters
    ----------
    returns:
        Polars DataFrame or Series.

    Returns
    -------
    pl.DataFrame | pl.Series
        The input unchanged.
    """
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
    """Return the conventional number of periods per year for a given frequency.

    Parameters
    ----------
    frequency:
        One of ``"D"`` (daily, 252), ``"W"`` (weekly, 52), ``"M"`` (monthly, 12),
        ``"Q"`` (quarterly, 4), ``"A"`` / ``"Y"`` (annual, 1).

    Returns
    -------
    int
        Conventional periods per year.

    Raises
    ------
    ValueError
        If *frequency* is not recognized.
    """
    key = frequency.upper()
    if key not in _FREQUENCY_PERIODS:
        raise ValueError(
            f"Unknown frequency '{frequency}'. "
            f"Valid options: {list(_FREQUENCY_PERIODS.keys())}"
        )
    return _FREQUENCY_PERIODS[key]


def annual_to_periodic(rate: float, *, periods_per_year: float) -> float:
    """Convert an annualized rate to a per-period equivalent.

    Uses geometric conversion: ``(1 + annual_rate)^(1/periods_per_year) - 1``.

    Parameters
    ----------
    rate:
        Annualized rate (e.g. 0.05 for 5%).
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    float
        Per-period rate.
    """
    if periods_per_year <= 0:
        raise ValueError(f"'periods_per_year' must be positive; got {periods_per_year}")
    return (1.0 + rate) ** (1.0 / periods_per_year) - 1.0


def periodic_to_annual(rate: float, *, periods_per_year: float) -> float:
    """Convert a per-period rate to an annualized equivalent.

    Uses geometric conversion: ``(1 + periodic_rate)^periods_per_year - 1``.

    Parameters
    ----------
    rate:
        Per-period rate (e.g. 0.004 for ~0.4%/month).
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    float
        Annualized rate.
    """
    if periods_per_year <= 0:
        raise ValueError(f"'periods_per_year' must be positive; got {periods_per_year}")
    return (1.0 + rate) ** periods_per_year - 1.0
