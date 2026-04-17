"""Rolling (windowed) versions of core metrics.

Inputs: pl.Series / np.ndarray (NaNs propagated within each window — not dropped).
Outputs: pl.Series length-aligned to input, leading `window - 1` values null, cast to Float32.
`window` is an int or a Polars duration string (duration requires a time index).
Internal computation runs in Float64.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from ruin._internal.validate import FLOAT_DTYPE, INTERNAL_FLOAT_DTYPE, ReturnInput
from ruin.activity import profit_factor as _pf
from ruin.distribution import autocorrelation as _autocorr
from ruin.distribution import excess_kurtosis as _kurtosis
from ruin.distribution import skewness as _skewness
from ruin.drawdown import max_drawdown as _mdd


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
    """Resolve our `min_periods` kwarg to Polars' `min_samples`."""
    if min_periods is not None:
        if min_periods < 1:
            raise ValueError(f"'min_periods' must be >= 1; got {min_periods}")
        return min_periods
    return window if isinstance(window, int) else 1


def _require_int_window(window: int | str, func_name: str) -> int:
    """Validate that the window is an integer >= 1."""
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
    """Apply a scalar metric over rolling windows, returning a Float32 Series.

    Used for metrics not expressible as a native Polars rolling expression (skew, kurtosis,
    path-dependent max drawdown). Windows with fewer than `min_periods` valid obs emit null.
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
    """Rolling standard deviation. `min_periods` defaults to `window` for int windows."""
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
    """Rolling downside deviation relative to `threshold`."""
    s = _ensure_series(returns)
    downside = (s - threshold).clip(upper_bound=0.0)
    out = (
        (downside**2).rolling_mean(window_size=window, min_samples=_mp(min_periods, window)).sqrt()
    )
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
    """Rolling annualized Sharpe ratio."""
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
    """Rolling annualized Sortino ratio. `threshold` defaults to `risk_free`."""
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
    """Rolling beta vs. benchmark. Integer window only."""
    r = _ensure_series(returns)
    b = _ensure_series(benchmark)
    _require_matching_lengths(r, b)
    _require_int_window(window, "rolling_beta")
    mp = _mp(min_periods, window)
    df = pl.DataFrame({"r": r, "b": b})
    roll_cov = df.select(pl.rolling_cov("r", "b", window_size=window, min_samples=mp)).to_series()
    roll_var = df["b"].rolling_var(window_size=window, min_samples=mp)
    return (roll_cov / roll_var).cast(FLOAT_DTYPE).rename("rolling_beta")


def rolling_correlation(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    window: int | str,
    min_periods: int | None = None,
) -> pl.Series:
    """Rolling Pearson correlation with benchmark. Integer window only."""
    r = _ensure_series(returns)
    b = _ensure_series(benchmark)
    _require_matching_lengths(r, b)
    _require_int_window(window, "rolling_correlation")
    mp = _mp(min_periods, window)
    df = pl.DataFrame({"r": r, "b": b})
    out = df.select(pl.rolling_corr("r", "b", window_size=window, min_samples=mp)).to_series()
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
    """Rolling annualized tracking error (std of `returns - benchmark`)."""
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
    """Rolling annualized Jensen's alpha."""
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
    """Rolling skewness. Integer window only."""
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
    """Rolling excess kurtosis. Integer window only."""
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
    """Rolling lag-`k` autocorrelation. Integer window only."""
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
    """Rolling max drawdown (non-positive), recomputed from scratch per window. Integer window."""
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
    """Rolling hit rate (fraction of periods > threshold), in [0, 1]."""
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
    """Rolling profit factor. Integer window only."""
    s = _ensure_series(returns)
    window_int = _require_int_window(window, "rolling_profit_factor")
    mp = min_periods if min_periods is not None else window_int

    def _apply(series: pl.Series) -> float:
        return _pf(series, threshold=threshold)

    return _window_apply(s, window_int, mp, _apply, name="rolling_profit_factor")
