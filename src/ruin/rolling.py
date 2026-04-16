"""Rolling (windowed) versions of core metrics.

All functions accept ``pl.Series`` or ``np.ndarray``. They return a Polars ``Series``
aligned to the input with the leading ``window - 1`` values set to null.

``window`` can be an integer (number of periods) or a Polars duration string (e.g. ``"30d"``),
subject to the Series having an associated time index when a string is used. In practice,
integer windows are the common case.

NaN values in the input are propagated within each window. Outputs are cast to
``pl.Float32`` for memory efficiency; internal computation runs in ``pl.Float64``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from ruin._internal.validate import FLOAT_DTYPE, INTERNAL_FLOAT_DTYPE, ReturnInput


def _ensure_series(returns: ReturnInput) -> pl.Series:
    """Coerce to Series without NaN-dropping (rolling functions need aligned output)."""
    if isinstance(returns, pl.Series):
        return returns.cast(INTERNAL_FLOAT_DTYPE)
    if isinstance(returns, np.ndarray):
        if returns.ndim != 1:
            raise ValueError(f"Array must be 1-D; got shape {returns.shape}")
        return pl.Series("returns", returns, dtype=INTERNAL_FLOAT_DTYPE)
    if isinstance(returns, pl.DataFrame):
        if returns.width != 1:
            raise ValueError("Pass a single-column DataFrame for rolling functions.")
        return returns.to_series(0).cast(INTERNAL_FLOAT_DTYPE)
    raise TypeError(f"Unsupported type: {type(returns).__name__}")


def _mp(min_periods: int | None, window: int | str) -> int | None:
    """Resolve min_periods (our kwarg) to the effective value passed to Polars (min_samples)."""
    if min_periods is not None:
        if min_periods < 1:
            raise ValueError(f"'min_periods' must be >= 1; got {min_periods}")
        return min_periods
    return window if isinstance(window, int) else 1


def _require_int_window(window: int | str, func_name: str) -> int:
    """Validate that a rolling function received an integer window."""
    if not isinstance(window, int):
        raise TypeError(f"{func_name} requires an integer window; got {type(window).__name__}.")
    if window < 1:
        raise ValueError(f"{func_name} requires window >= 1; got {window}.")
    return window


def _require_matching_lengths(r: pl.Series, b: pl.Series) -> None:
    """Raise if returns and benchmark have different lengths."""
    if len(r) != len(b):
        raise ValueError(f"returns (len={len(r)}) and benchmark (len={len(b)}) must match.")


def _window_apply(
    s: pl.Series,
    window: int,
    min_periods: int,
    fn: Callable[[pl.Series], float],
    *,
    name: str,
) -> pl.Series:
    """Apply a scalar metric over rolling windows and return a Float32 Series.

    Used by rolling metrics that cannot be expressed as a single Polars rolling
    expression (e.g. skewness, kurtosis, path-dependent max drawdown). Windows
    with fewer than *min_periods* non-null values emit null.
    """
    n = len(s)
    result: list[float | None] = [None] * n
    for i in range(window - 1, n):
        w = s.slice(i - window + 1, window)
        valid = w.drop_nans().drop_nulls()
        if len(valid) < min_periods:
            continue
        try:
            result[i] = float(fn(valid))
        except (ValueError, ZeroDivisionError):
            result[i] = None
    return pl.Series(name, result, dtype=FLOAT_DTYPE)


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
    out = s.rolling_std(window_size=window, min_samples=_mp(min_periods, window), ddof=ddof)
    return out.cast(FLOAT_DTYPE).rename("rolling_volatility")


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
    out = (downside**2).rolling_mean(
        window_size=window, min_samples=_mp(min_periods, window)
    ).sqrt()
    return out.cast(FLOAT_DTYPE).rename("rolling_downside_deviation")


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
    return (ann_mean / ann_std).cast(FLOAT_DTYPE).rename("rolling_sharpe")


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
    return (ann_mean / ann_dd).cast(FLOAT_DTYPE).rename("rolling_sortino")


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
    _require_matching_lengths(r, b)
    _require_int_window(window, "rolling_beta")
    mp = _mp(min_periods, window)
    df = pl.DataFrame({"r": r, "b": b})
    roll_cov = df.select(
        pl.rolling_cov("r", "b", window_size=window, min_samples=mp)
    ).to_series()
    roll_var = df["b"].rolling_var(window_size=window, min_samples=mp)
    return (roll_cov / roll_var).cast(FLOAT_DTYPE).rename("rolling_beta")


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
    _require_matching_lengths(r, b)
    _require_int_window(window, "rolling_correlation")
    mp = _mp(min_periods, window)
    df = pl.DataFrame({"r": r, "b": b})
    out = df.select(
        pl.rolling_corr("r", "b", window_size=window, min_samples=mp)
    ).to_series()
    return out.cast(FLOAT_DTYPE).rename("rolling_correlation")


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
    _require_matching_lengths(r, b)
    mp = _mp(min_periods, window)
    active = r - b
    out = active.rolling_std(window_size=window, min_samples=mp, ddof=ddof) * (
        periods_per_year**0.5
    )
    return out.cast(FLOAT_DTYPE).rename("rolling_tracking_error")


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
    _require_matching_lengths(r, b)
    mp = _mp(min_periods, window)
    r_exc = r - risk_free
    b_exc = b - risk_free
    roll_mean_r = r_exc.rolling_mean(window_size=window, min_samples=mp)
    roll_mean_b = b_exc.rolling_mean(window_size=window, min_samples=mp)
    roll_beta_series = rolling_beta(r, b, window=window, min_periods=min_periods).cast(
        INTERNAL_FLOAT_DTYPE
    )
    out = (roll_mean_r - roll_beta_series * roll_mean_b) * periods_per_year
    return out.cast(FLOAT_DTYPE).rename("rolling_alpha")


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
    window_int = _require_int_window(window, "rolling_skewness")
    mp = min_periods if min_periods is not None else window_int
    return _window_apply(s, window_int, mp, _skewness, name="rolling_skewness")


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
    window_int = _require_int_window(window, "rolling_excess_kurtosis")
    mp = min_periods if min_periods is not None else window_int
    return _window_apply(s, window_int, mp, _kurtosis, name="rolling_excess_kurtosis")


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
    window_int = _require_int_window(window, "rolling_autocorrelation")
    mp = min_periods if min_periods is not None else window_int

    def _apply(series: pl.Series) -> float:
        return _autocorr(series, lag=lag)

    return _window_apply(s, window_int, mp, _apply, name="rolling_autocorrelation")


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
    window_int = _require_int_window(window, "rolling_max_drawdown")
    mp = min_periods if min_periods is not None else window_int
    return _window_apply(s, window_int, mp, _mdd, name="rolling_max_drawdown")


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
    wins = (s > threshold).cast(INTERNAL_FLOAT_DTYPE)
    out = wins.rolling_mean(window_size=window, min_samples=mp)
    return out.cast(FLOAT_DTYPE).rename("rolling_hit_rate")


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
    window_int = _require_int_window(window, "rolling_profit_factor")
    mp = min_periods if min_periods is not None else window_int

    def _apply(series: pl.Series) -> float:
        return _pf(series, threshold=threshold)

    return _window_apply(s, window_int, mp, _apply, name="rolling_profit_factor")
