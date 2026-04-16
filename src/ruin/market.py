"""Benchmark-relative market metrics.

All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame``.
NaN values are dropped before computation. When both inputs are bare Series/arrays,
equal length is required and the caller is trusted for alignment.
"""

from __future__ import annotations

import polars as pl

from ruin._internal.validate import ReturnInput, align_benchmark, require_minimum_length


def beta(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Market beta: covariance(returns, benchmark) / variance(benchmark).

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.

    Returns
    -------
    float
        Beta coefficient. ``float('nan')`` if benchmark variance is zero.
    """
    r, b = align_benchmark(returns, benchmark)
    require_minimum_length(r, 2, "beta")
    df = pl.DataFrame({"r": r, "b": b})
    cov_val = float(df.select(pl.corr("r", "b")).item()) * float(r.std(ddof=1)) * float(b.std(ddof=1))  # type: ignore[arg-type]
    var_b = float(b.var(ddof=1))  # type: ignore[arg-type]
    if var_b == 0.0:
        return float("nan")
    return cov_val / var_b


def _cov_manual(a: pl.Series, b: pl.Series, ddof: int = 1) -> float:
    """Compute covariance without relying on Polars cov method for filtered subsets."""
    n = len(a)
    if n - ddof <= 0:
        return float("nan")
    mean_a = float(a.mean())  # type: ignore[arg-type]
    mean_b = float(b.mean())  # type: ignore[arg-type]
    return float(((a - mean_a) * (b - mean_b)).sum()) / (n - ddof)


def downside_beta(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Beta computed only on periods where the benchmark return is negative.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.

    Returns
    -------
    float
        Downside beta. ``float('nan')`` if no negative benchmark periods or
        benchmark variance is zero in that subset.
    """
    r, b = align_benchmark(returns, benchmark)
    mask = b < 0.0
    r_down = r.filter(mask)
    b_down = b.filter(mask)
    if len(b_down) < 2:
        return float("nan")
    cov_val = _cov_manual(r_down, b_down, ddof=1)
    var_b = _cov_manual(b_down, b_down, ddof=1)
    if var_b == 0.0:
        return float("nan")
    return cov_val / var_b


def upside_beta(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Beta computed only on periods where the benchmark return is positive.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.

    Returns
    -------
    float
        Upside beta. ``float('nan')`` if no positive benchmark periods or
        benchmark variance is zero in that subset.
    """
    r, b = align_benchmark(returns, benchmark)
    mask = b > 0.0
    r_up = r.filter(mask)
    b_up = b.filter(mask)
    if len(b_up) < 2:
        return float("nan")
    cov_val = _cov_manual(r_up, b_up, ddof=1)
    var_b = _cov_manual(b_up, b_up, ddof=1)
    if var_b == 0.0:
        return float("nan")
    return cov_val / var_b


def alpha(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    risk_free: float = 0.0,
    periods_per_year: float,
) -> float:
    """Annualized Jensen's alpha.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.
    risk_free:
        Per-period risk-free rate. Default 0.0.
    periods_per_year:
        Number of periods in a year for annualization.

    Returns
    -------
    float
        Annualized alpha = annualized(r_excess) - beta * annualized(b_excess).
    """
    r, b = align_benchmark(returns, benchmark)
    beta_val = beta(r, b)
    r_excess = r - risk_free
    b_excess = b - risk_free
    ann_r = float(r_excess.mean()) * periods_per_year  # type: ignore[operator]
    ann_b = float(b_excess.mean()) * periods_per_year  # type: ignore[operator]
    return ann_r - beta_val * ann_b


def tracking_error(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Annualized tracking error (std of active returns).

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.
    periods_per_year:
        Number of periods in a year.
    ddof:
        Delta degrees of freedom. Default 1.

    Returns
    -------
    float
        Annualized standard deviation of (returns - benchmark).
    """
    r, b = align_benchmark(returns, benchmark)
    active = r - b
    return float(active.std(ddof=ddof)) * (periods_per_year**0.5)  # type: ignore[arg-type]


def correlation(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Pearson correlation between returns and benchmark.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].
    """
    r, b = align_benchmark(returns, benchmark)
    require_minimum_length(r, 2, "correlation")
    df = pl.DataFrame({"r": r, "b": b})
    return float(df.select(pl.corr("r", "b")).item())


def up_capture(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Up-market capture ratio (geometric compounding).

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.

    Returns
    -------
    float
        Compounded return of strategy during benchmark-up periods divided by
        compounded benchmark return during those periods.
        ``float('nan')`` if no up periods.
    """
    r, b = align_benchmark(returns, benchmark)
    mask = b > 0.0
    r_up = r.filter(mask)
    b_up = b.filter(mask)
    if len(b_up) == 0:
        return float("nan")
    strategy_geo = float((1.0 + r_up).product()) - 1.0
    bench_geo = float((1.0 + b_up).product()) - 1.0
    if bench_geo == 0.0:
        return float("nan")
    return strategy_geo / bench_geo


def down_capture(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Down-market capture ratio (geometric compounding).

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    benchmark:
        Benchmark return series. Must be same length after NaN removal.

    Returns
    -------
    float
        Compounded return of strategy during benchmark-down periods divided by
        compounded benchmark return during those periods.
        ``float('nan')`` if no down periods.
    """
    r, b = align_benchmark(returns, benchmark)
    mask = b < 0.0
    r_down = r.filter(mask)
    b_down = b.filter(mask)
    if len(b_down) == 0:
        return float("nan")
    strategy_geo = float((1.0 + r_down).product()) - 1.0
    bench_geo = float((1.0 + b_down).product()) - 1.0
    if bench_geo == 0.0:
        return float("nan")
    return strategy_geo / bench_geo
