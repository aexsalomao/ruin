"""Summary report — the single allowed bundling function. Pure composition of module-level metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from ruin._internal.validate import FLOAT_DTYPE, ReturnInput, check_nan_strict, to_series
from ruin.activity import (
    average_loss,
    average_win,
    best_period,
    hit_rate,
    longest_losing_streak,
    longest_winning_streak,
    profit_factor,
    win_loss_ratio,
    worst_period,
)
from ruin.distribution import autocorrelation, excess_kurtosis, jarque_bera, skewness
from ruin.drawdown import (
    average_drawdown,
    max_drawdown,
    max_drawdown_duration,
    recovery_time,
    time_underwater,
)
from ruin.market import (
    alpha,
    beta,
    correlation,
    down_capture,
    tracking_error,
    up_capture,
)
from ruin.ratios import (
    calmar_ratio,
    information_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
)
from ruin.returns import annualize_return, cagr, total_return
from ruin.tail import conditional_value_at_risk, value_at_risk
from ruin.volatility import annualize_volatility, downside_deviation, volatility


def summary(
    returns: ReturnInput,
    benchmark: ReturnInput | None = None,
    *,
    risk_free: float = 0.0,
    periods_per_year: float,
    strict: bool = False,
) -> pl.DataFrame:
    """Every scalar metric as a Polars DataFrame — one row per return stream.

    DataFrame input produces one row per column. Benchmark-relative columns (alpha, beta, etc.)
    are null when no benchmark is given. `strict=True` raises on NaN/null input instead of dropping.
    """
    if strict:
        check_nan_strict(returns)
        if benchmark is not None:
            check_nan_strict(benchmark, name="benchmark")

    # Handle DataFrame input (multi-strategy)
    if isinstance(returns, pl.DataFrame):
        rows: list[dict[str, Any]] = []
        for col in returns.columns:
            r_col = returns[col]
            b_col: pl.Series | None
            if isinstance(benchmark, pl.DataFrame) and col in benchmark.columns:
                b_col = benchmark[col]
            elif isinstance(benchmark, pl.Series):
                b_col = benchmark
            else:
                b_col = None
            row = _compute_row(
                r_col,
                b_col,
                risk_free=risk_free,
                periods_per_year=periods_per_year,
                col_name=col,
            )
            rows.append(row)
        return _cast_float_columns(pl.from_dicts(rows))

    r = to_series(returns)
    b = to_series(benchmark) if benchmark is not None else None
    row = _compute_row(r, b, risk_free=risk_free, periods_per_year=periods_per_year)
    return _cast_float_columns(pl.from_dicts([row]))


def _cast_float_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast Float64 columns to public FLOAT_DTYPE (Float32)."""
    float_cols = {
        col: FLOAT_DTYPE
        for col, dtype in zip(df.columns, df.dtypes, strict=True)
        if dtype == pl.Float64
    }
    return df.cast(float_cols) if float_cols else df


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call `fn`; return NaN on "metric undefined" errors (too few obs, zero variance). Other errors propagate."""
    try:
        result = fn(*args, **kwargs)
    except (ValueError, ZeroDivisionError):
        return float("nan")
    if result is None:
        return float("nan")
    return result


def _compute_row(
    r: pl.Series,
    b: pl.Series | None,
    *,
    risk_free: float,
    periods_per_year: float,
    col_name: str = "returns",
) -> dict[str, Any]:
    """All metrics for one return stream as a `{name: value}` dict."""
    row: dict[str, Any] = {"name": col_name}

    # Returns
    row["total_return"] = _safe(total_return, r)
    row["cagr"] = _safe(cagr, r, periods_per_year=periods_per_year)
    row["annualized_return_arithmetic"] = _safe(
        annualize_return, r, periods_per_year=periods_per_year, method="arithmetic"
    )

    # Volatility
    row["volatility"] = _safe(volatility, r)
    row["annualized_volatility"] = _safe(annualize_volatility, r, periods_per_year=periods_per_year)
    row["downside_deviation"] = _safe(downside_deviation, r, threshold=risk_free)

    # Drawdown
    row["max_drawdown"] = _safe(max_drawdown, r)
    row["average_drawdown"] = _safe(average_drawdown, r)
    row["max_drawdown_duration"] = _safe(max_drawdown_duration, r)
    row["recovery_time"] = _safe(recovery_time, r)
    row["time_underwater"] = _safe(time_underwater, r)

    # Ratios
    row["sharpe_ratio"] = _safe(
        sharpe_ratio, r, risk_free=risk_free, periods_per_year=periods_per_year
    )
    row["sortino_ratio"] = _safe(
        sortino_ratio, r, risk_free=risk_free, periods_per_year=periods_per_year
    )
    row["calmar_ratio"] = _safe(calmar_ratio, r, periods_per_year=periods_per_year)
    row["omega_ratio"] = _safe(omega_ratio, r, threshold=risk_free)

    # Tail
    row["var_95_historical"] = _safe(value_at_risk, r, confidence=0.95, method="historical")
    row["cvar_95_historical"] = _safe(
        conditional_value_at_risk, r, confidence=0.95, method="historical"
    )
    row["var_95_parametric"] = _safe(value_at_risk, r, confidence=0.95, method="parametric")
    row["cvar_95_parametric"] = _safe(
        conditional_value_at_risk, r, confidence=0.95, method="parametric"
    )

    # Distribution
    row["skewness"] = _safe(skewness, r)
    row["excess_kurtosis"] = _safe(excess_kurtosis, r)
    row["autocorrelation_lag1"] = _safe(autocorrelation, r, lag=1)
    jb = _safe(jarque_bera, r)
    row["jarque_bera_stat"] = jb.statistic if hasattr(jb, "statistic") else float("nan")
    row["jarque_bera_pvalue"] = jb.p_value if hasattr(jb, "p_value") else float("nan")

    # Activity
    row["hit_rate"] = _safe(hit_rate, r, threshold=risk_free)
    row["average_win"] = _safe(average_win, r, threshold=risk_free)
    row["average_loss"] = _safe(average_loss, r, threshold=risk_free)
    row["win_loss_ratio"] = _safe(win_loss_ratio, r, threshold=risk_free)
    row["profit_factor"] = _safe(profit_factor, r, threshold=risk_free)
    row["best_period"] = _safe(best_period, r)
    row["worst_period"] = _safe(worst_period, r)
    row["longest_winning_streak"] = _safe(longest_winning_streak, r, threshold=risk_free)
    row["longest_losing_streak"] = _safe(longest_losing_streak, r, threshold=risk_free)

    # Benchmark-relative (null if no benchmark)
    if b is not None:
        row["beta"] = _safe(beta, r, b)
        row["alpha"] = _safe(alpha, r, b, risk_free=risk_free, periods_per_year=periods_per_year)
        row["correlation"] = _safe(correlation, r, b)
        row["tracking_error"] = _safe(tracking_error, r, b, periods_per_year=periods_per_year)
        row["information_ratio"] = _safe(information_ratio, r, b, periods_per_year=periods_per_year)
        row["treynor_ratio"] = _safe(
            treynor_ratio, r, b, risk_free=risk_free, periods_per_year=periods_per_year
        )
        row["up_capture"] = _safe(up_capture, r, b)
        row["down_capture"] = _safe(down_capture, r, b)
    else:
        for col in [
            "beta",
            "alpha",
            "correlation",
            "tracking_error",
            "information_ratio",
            "treynor_ratio",
            "up_capture",
            "down_capture",
        ]:
            row[col] = None

    return row
