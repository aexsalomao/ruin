"""Benchmark-relative market metrics. NaNs dropped; bare Series/arrays must already be aligned."""

from __future__ import annotations

import polars as pl

from ruin._internal.validate import ReturnInput, align_benchmark, require_minimum_length


def beta(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Market beta: `cov(r, b) / var(b)`. NaN if benchmark variance is zero."""
    r, b = align_benchmark(returns, benchmark)
    require_minimum_length(r, 2, "beta")
    var_b = float(b.var(ddof=1))
    if var_b == 0.0:
        return float("nan")
    return _cov_manual(r, b, ddof=1) / var_b


def _cov_manual(a: pl.Series, b: pl.Series, ddof: int = 1) -> float:
    """Covariance without relying on Polars cov() for filtered subsets."""
    n = len(a)
    if n - ddof <= 0:
        return float("nan")
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    return float(((a - mean_a) * (b - mean_b)).sum()) / (n - ddof)


def downside_beta(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Beta on periods where benchmark < 0. NaN if fewer than 2 such periods or zero variance."""
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
    """Beta on periods where benchmark > 0. NaN if fewer than 2 such periods or zero variance."""
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
    """Annualized Jensen's alpha: `ann(r_excess) - beta * ann(b_excess)`."""
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
    """Annualized std of active returns (`returns - benchmark`)."""
    r, b = align_benchmark(returns, benchmark)
    active = r - b
    return float(active.std(ddof=ddof)) * (periods_per_year**0.5)


def correlation(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Pearson correlation in [-1, 1]."""
    r, b = align_benchmark(returns, benchmark)
    require_minimum_length(r, 2, "correlation")
    df = pl.DataFrame({"r": r, "b": b})
    return float(df.select(pl.corr("r", "b")).item())


def up_capture(returns: ReturnInput, benchmark: ReturnInput) -> float:
    """Up-market capture (geometric): strategy compound / benchmark compound, over b > 0 periods."""
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
    """Down-market capture (geometric): strategy compound / bench compound, over b < 0 periods."""
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
