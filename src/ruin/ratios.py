"""Risk-adjusted performance ratios. Inputs: pl.Series / np.ndarray / pl.DataFrame; NaNs dropped."""

from __future__ import annotations

from ruin._internal.validate import (
    ReturnInput,
    align_benchmark,
    require_minimum_length,
    to_series,
)
from ruin.drawdown import max_drawdown
from ruin.market import beta as _compute_beta
from ruin.returns import annualize_return
from ruin.volatility import downside_deviation


def sharpe_ratio(
    returns: ReturnInput,
    *,
    risk_free: float = 0.0,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Annualized Sharpe: `ann_excess_return / ann_volatility`. `risk_free` is per-period."""
    r = to_series(returns)
    require_minimum_length(r, ddof + 1, "sharpe_ratio")
    excess = r - risk_free
    ann_excess = float(excess.mean()) * periods_per_year  # type: ignore[operator]
    ann_vol = float(excess.std(ddof=ddof)) * (periods_per_year**0.5)
    if ann_vol == 0.0:
        return float("nan")
    return ann_excess / ann_vol


def sortino_ratio(
    returns: ReturnInput,
    *,
    risk_free: float = 0.0,
    threshold: float | None = None,
    periods_per_year: float,
) -> float:
    """Annualized Sortino: `ann_excess_return / ann_downside_deviation`. `threshold` defaults to rf.

    Downside deviation uses `ddof=0` with all periods in the denominator (not just downside).
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "sortino_ratio")
    mar = risk_free if threshold is None else threshold
    excess_mean = float((r - risk_free).mean()) * periods_per_year  # type: ignore[operator]
    dd = downside_deviation(r, threshold=mar, ddof=0)
    ann_dd = dd * (periods_per_year**0.5)
    if ann_dd == 0.0:
        return float("nan")
    return excess_mean / ann_dd


def calmar_ratio(returns: ReturnInput, *, periods_per_year: float) -> float:
    """Calmar: CAGR / |max drawdown|. NaN if max drawdown is zero."""
    r = to_series(returns)
    require_minimum_length(r, 1, "calmar_ratio")
    ann_ret = annualize_return(r, periods_per_year=periods_per_year)
    mdd = max_drawdown(r)
    if mdd == 0.0:
        return float("nan")
    return ann_ret / abs(mdd)


def information_ratio(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Annualized IR: `ann_active_return / ann_tracking_error`."""
    r, b = align_benchmark(returns, benchmark)
    active = r - b
    ann_active = float(active.mean()) * periods_per_year  # type: ignore[operator]
    ann_te = float(active.std(ddof=ddof)) * (periods_per_year**0.5)
    if ann_te == 0.0:
        return float("nan")
    return ann_active / ann_te


def treynor_ratio(
    returns: ReturnInput,
    benchmark: ReturnInput,
    *,
    risk_free: float = 0.0,
    periods_per_year: float,
) -> float:
    """Annualized Treynor: `ann_excess_return / beta`."""
    r, b = align_benchmark(returns, benchmark)
    ann_excess = float((r - risk_free).mean()) * periods_per_year  # type: ignore[operator]
    beta_val = _compute_beta(r, b)
    if beta_val == 0.0:
        return float("nan")
    return ann_excess / beta_val


def omega_ratio(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Omega: `sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))`. NaN if no downside.

    Omega > 1 iff `mean(r) > threshold` for any distribution with finite mean.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "omega_ratio")
    gains = (r - threshold).clip(lower_bound=0.0).sum()
    losses = (threshold - r).clip(lower_bound=0.0).sum()
    if losses == 0.0:
        return float("nan")
    return float(gains) / float(losses)
