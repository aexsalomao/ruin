"""Rolling (windowed) versions of core metrics.

All functions accept ``pl.Series`` or ``np.ndarray``. They return a Polars ``Series``
aligned to the input with the leading ``window - 1`` values set to null.

``window`` can be an integer (number of periods) or a Polars duration string (e.g. ``"30d"``),
subject to the Series having an associated time index when a string is used. In practice,
integer windows are the common case.

NaN values in the input are propagated within each window.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ruin._internal.validate import ReturnInput


def _ensure_series(returns: ReturnInput) -> pl.Series:
    """Coerce to Series without NaN-dropping (rolling functions need aligned output)."""
    if isinstance(returns, pl.Series):
        return returns.cast(pl.Float64)
    if isinstance(returns, np.ndarray):
        if returns.ndim != 1:
            raise ValueError(f"Array must be 1-D; got shape {returns.shape}")
        return pl.Series("returns", returns, dtype=pl.Float64)
    if isinstance(returns, pl.DataFrame):
        if returns.width != 1:
            raise ValueError("Pass a single-column DataFrame for rolling functions.")
        return returns.to_series(0).cast(pl.Float64)
    raise TypeError(f"Unsupported type: {type(returns).__name__}")


def _mp(min_periods: int | None, window: int | str) -> int | None:
    """Resolve min_periods (our kwarg) to the effective value passed to Polars (min_samples)."""
    if min_periods is not None:
        return min_periods
    return window if isinstance(window, int) else 1


def rolling_volatility(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pl.Series:
    """Rolling standard deviation of returns.

    Parameters
    ----------
    returns:
        Periodic return series (aligned; NaNs not dropped).
    window:
        Rolling window size in periods (int) or Polars duration string.
    min_periods:
        Minimum non-null observations required. Defaults to *window* for int windows.
    ddof:
        Delta degrees of freedom. Default 1.

    Returns
    -------
    pl.Series
        Rolling volatility, length-aligned to input. Leading nulls for incomplete windows.
    """
    s = _ensure_series(returns)
    return s.rolling_std(window_size=window, min_samples=_mp(min_periods, window), ddof=ddof)


def rolling_downside_deviation(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    threshold: float = 0.0,
) -> pl.Series:
    """Rolling downside deviation.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    pl.Series
        Rolling downside deviation aligned to input.
    """
    s = _ensure_series(returns)
    downside = (s - threshold).clip(upper_bound=0.0)
    return (downside**2).rolling_mean(window_size=window, min_samples=_mp(min_periods, window)).sqrt()


def rolling_sharpe(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    risk_free: float = 0.0,
    periods_per_year: float,
    ddof: int = 1,
) -> pl.Series:
    """Rolling annualized Sharpe ratio.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.
    risk_free:
        Per-period risk-free rate. Default 0.0.
    periods_per_year:
        Used for annualization.
    ddof:
        Delta degrees of freedom. Default 1.

    Returns
    -------
    pl.Series
        Rolling Sharpe ratio aligned to input.
    """
    s = _ensure_series(returns)
    mp = _mp(min_periods, window)
    excess = s - risk_free
    roll_mean = excess.rolling_mean(window_size=window, min_samples=mp)
    roll_std = excess.rolling_std(window_size=window, min_samples=mp, ddof=ddof)
    ann_mean = roll_mean * periods_per_year
    ann_std = roll_std * (periods_per_year**0.5)
    return ann_mean / ann_std


def rolling_sortino(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    risk_free: float = 0.0,
    threshold: float | None = None,
    periods_per_year: float,
) -> pl.Series:
    """Rolling annualized Sortino ratio.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.
    risk_free:
        Per-period risk-free rate. Default 0.0.
    threshold:
        MAR for downside deviation. Defaults to ``risk_free``.
    periods_per_year:
        Used for annualization.

    Returns
    -------
    pl.Series
        Rolling Sortino ratio aligned to input.
    """
    s = _ensure_series(returns)
    mp = _mp(min_periods, window)
    mar = risk_free if threshold is None else threshold
    excess = s - risk_free
    roll_mean = excess.rolling_mean(window_size=window, min_samples=mp)
    downside = (s - mar).clip(upper_bound=0.0)
    roll_dd = (downside**2).rolling_mean(window_size=window, min_samples=mp).sqrt()
    ann_mean = roll_mean * periods_per_year
    ann_dd = roll_dd * (periods_per_year**0.5)
    return ann_mean / ann_dd


def rolling_beta(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
) -> pl.Series:
    """Rolling beta versus a benchmark.

    Parameters
    ----------
    returns:
        Periodic return series.
    benchmark:
        Benchmark return series (same length).
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.

    Returns
    -------
    pl.Series
        Rolling beta aligned to input.
    """
    r = _ensure_series(returns)
    b = _ensure_series(benchmark)
    if len(r) != len(b):
        raise ValueError(f"returns (len={len(r)}) and benchmark (len={len(b)}) must match.")
    if not isinstance(window, int):
        raise TypeError("rolling_beta requires an integer window.")
    mp = _mp(min_periods, window)
    df = pl.DataFrame({"r": r, "b": b})
    roll_cov = df.select(
        pl.rolling_cov("r", "b", window_size=window, min_samples=mp)
    ).to_series()
    roll_var = df["b"].rolling_var(window_size=window, min_samples=mp)
    return roll_cov / roll_var


def rolling_correlation(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
) -> pl.Series:
    """Rolling Pearson correlation with a benchmark.

    Parameters
    ----------
    returns:
        Periodic return series.
    benchmark:
        Benchmark return series.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.

    Returns
    -------
    pl.Series
        Rolling correlation aligned to input.
    """
    r = _ensure_series(returns)
    b = _ensure_series(benchmark)
    if len(r) != len(b):
        raise ValueError(f"returns (len={len(r)}) and benchmark (len={len(b)}) must match.")
    if not isinstance(window, int):
        raise TypeError("rolling_correlation requires an integer window.")
    mp = _mp(min_periods, window)
    df = pl.DataFrame({"r": r, "b": b})
    return df.select(
        pl.rolling_corr("r", "b", window_size=window, min_samples=mp)
    ).to_series()


def rolling_tracking_error(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    periods_per_year: float,
    ddof: int = 1,
) -> pl.Series:
    """Rolling annualized tracking error.

    Parameters
    ----------
    returns:
        Periodic return series.
    benchmark:
        Benchmark return series.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.
    periods_per_year:
        Used for annualization.
    ddof:
        Delta degrees of freedom. Default 1.

    Returns
    -------
    pl.Series
        Rolling tracking error aligned to input.
    """
    r = _ensure_series(returns)
    b = _ensure_series(benchmark)
    if len(r) != len(b):
        raise ValueError(f"returns (len={len(r)}) and benchmark (len={len(b)}) must match.")
    mp = _mp(min_periods, window)
    active = r - b
    return active.rolling_std(window_size=window, min_samples=mp, ddof=ddof) * (periods_per_year**0.5)


def rolling_alpha(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    risk_free: float = 0.0,
    periods_per_year: float,
) -> pl.Series:
    """Rolling annualized Jensen's alpha.

    Parameters
    ----------
    returns:
        Periodic return series.
    benchmark:
        Benchmark return series.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.
    risk_free:
        Per-period risk-free rate.
    periods_per_year:
        Used for annualization.

    Returns
    -------
    pl.Series
        Rolling alpha aligned to input.
    """
    r = _ensure_series(returns)
    b = _ensure_series(benchmark)
    if len(r) != len(b):
        raise ValueError(f"returns (len={len(r)}) and benchmark (len={len(b)}) must match.")
    mp = _mp(min_periods, window)
    r_exc = r - risk_free
    b_exc = b - risk_free
    roll_mean_r = r_exc.rolling_mean(window_size=window, min_samples=mp)
    roll_mean_b = b_exc.rolling_mean(window_size=window, min_samples=mp)
    roll_beta = rolling_beta(r, b, window=window, min_periods=min_periods)
    return (roll_mean_r - roll_beta * roll_mean_b) * periods_per_year


def rolling_skewness(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
) -> pl.Series:
    """Rolling skewness.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size (integer only).
    min_periods:
        Minimum observations required.

    Returns
    -------
    pl.Series
        Rolling skewness aligned to input.
    """
    from ruin.distribution import skewness as _skewness

    s = _ensure_series(returns)
    if not isinstance(window, int):
        raise TypeError("rolling_skewness requires an integer window.")
    mp = min_periods if min_periods is not None else window
    n = len(s)
    result: list[float | None] = [None] * n
    for i in range(window - 1, n):
        w = s.slice(i - window + 1, window)
        valid = w.drop_nans().drop_nulls()
        if len(valid) >= mp:
            try:
                result[i] = _skewness(valid)
            except Exception:
                result[i] = None
    return pl.Series("rolling_skewness", result, dtype=pl.Float64)


def rolling_excess_kurtosis(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
) -> pl.Series:
    """Rolling excess kurtosis.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size (integer only).
    min_periods:
        Minimum observations required.

    Returns
    -------
    pl.Series
        Rolling excess kurtosis aligned to input.
    """
    from ruin.distribution import excess_kurtosis as _kurtosis

    s = _ensure_series(returns)
    if not isinstance(window, int):
        raise TypeError("rolling_excess_kurtosis requires an integer window.")
    mp = min_periods if min_periods is not None else window
    n = len(s)
    result: list[float | None] = [None] * n
    for i in range(window - 1, n):
        w = s.slice(i - window + 1, window)
        valid = w.drop_nans().drop_nulls()
        if len(valid) >= mp:
            try:
                result[i] = _kurtosis(valid)
            except Exception:
                result[i] = None
    return pl.Series("rolling_excess_kurtosis", result, dtype=pl.Float64)


def rolling_autocorrelation(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    lag: int = 1,
) -> pl.Series:
    """Rolling lag-k autocorrelation.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size (integer only).
    min_periods:
        Minimum observations required.
    lag:
        Autocorrelation lag. Default 1.

    Returns
    -------
    pl.Series
        Rolling autocorrelation aligned to input.
    """
    from ruin.distribution import autocorrelation as _autocorr

    s = _ensure_series(returns)
    if not isinstance(window, int):
        raise TypeError("rolling_autocorrelation requires an integer window.")
    mp = min_periods if min_periods is not None else window
    n = len(s)
    result: list[float | None] = [None] * n
    for i in range(window - 1, n):
        w = s.slice(i - window + 1, window)
        valid = w.drop_nans().drop_nulls()
        if len(valid) >= mp:
            try:
                result[i] = _autocorr(valid, lag=lag)
            except Exception:
                result[i] = None
    return pl.Series("rolling_autocorrelation", result, dtype=pl.Float64)


def rolling_max_drawdown(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
) -> pl.Series:
    """Rolling maximum drawdown within each trailing window.

    Path-dependent: computed from scratch over each window.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size (integer only).
    min_periods:
        Minimum observations required.

    Returns
    -------
    pl.Series
        Rolling max drawdown (non-positive) aligned to input.
    """
    from ruin.drawdown import max_drawdown as _mdd

    s = _ensure_series(returns)
    if not isinstance(window, int):
        raise TypeError("rolling_max_drawdown requires an integer window.")
    mp = min_periods if min_periods is not None else window
    n = len(s)
    result: list[float | None] = [None] * n
    for i in range(window - 1, n):
        w = s.slice(i - window + 1, window)
        valid = w.drop_nans().drop_nulls()
        if len(valid) >= mp:
            try:
                result[i] = _mdd(valid)
            except Exception:
                result[i] = None
    return pl.Series("rolling_max_drawdown", result, dtype=pl.Float64)


def rolling_hit_rate(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    threshold: float = 0.0,
) -> pl.Series:
    """Rolling hit rate (fraction of periods above threshold).

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size.
    min_periods:
        Minimum observations required.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    pl.Series
        Rolling hit rate in [0, 1], aligned to input.
    """
    s = _ensure_series(returns)
    mp = _mp(min_periods, window)
    wins = (s > threshold).cast(pl.Float64)
    return wins.rolling_mean(window_size=window, min_samples=mp)


def rolling_profit_factor(
    returns: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
    threshold: float = 0.0,
) -> pl.Series:
    """Rolling profit factor.

    Parameters
    ----------
    returns:
        Periodic return series.
    window:
        Rolling window size (integer only).
    min_periods:
        Minimum observations required.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    pl.Series
        Rolling profit factor aligned to input.
    """
    from ruin.activity import profit_factor as _pf

    s = _ensure_series(returns)
    if not isinstance(window, int):
        raise TypeError("rolling_profit_factor requires an integer window.")
    mp = min_periods if min_periods is not None else window
    n = len(s)
    result: list[float | None] = [None] * n
    for i in range(window - 1, n):
        w = s.slice(i - window + 1, window)
        valid = w.drop_nans().drop_nulls()
        if len(valid) >= mp:
            try:
                result[i] = _pf(valid, threshold=threshold)
            except Exception:
                result[i] = None
    return pl.Series("rolling_profit_factor", result, dtype=pl.Float64)
