"""ruin — A Polars-first risk metrics library for quant hedge funds.

Curated public API. Import from submodules for the full interface.

Design principles
-----------------
- **Returns in, numbers out.** Operates on return series only.
- **Polars-first, minimal dependencies.** Runtime: polars + numpy only.
- **Pure functions.** Deterministic, no side effects, no hidden state.
- **One function, one metric, one return type.** Scalar → float, rolling → pl.Series.
- **Composition over convenience.** Call ``summary()`` or compose primitives yourself.
"""

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
from ruin.distribution import (
    JarqueBeraResult,
    autocorrelation,
    excess_kurtosis,
    jarque_bera,
    skewness,
)
from ruin.drawdown import (
    average_drawdown,
    drawdown_end,
    drawdown_series,
    drawdown_start,
    max_drawdown,
    max_drawdown_duration,
    recovery_time,
    time_underwater,
)
from ruin.inference import (
    bootstrap_metric,
    sharpe_confidence_interval,
    sharpe_standard_error,
)
from ruin.market import (
    alpha,
    beta,
    correlation,
    down_capture,
    downside_beta,
    tracking_error,
    up_capture,
    upside_beta,
)
from ruin.periods import (
    annual_to_periodic,
    mtd,
    periodic_to_annual,
    periods_per_year_for,
    qtd,
    since_inception,
    trailing,
    ytd,
)
from ruin.ratios import (
    calmar_ratio,
    information_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
)
from ruin.report import summary
from ruin.returns import annualize_return, cagr, from_prices, total_return
from ruin.rolling import (
    rolling_alpha,
    rolling_autocorrelation,
    rolling_beta,
    rolling_correlation,
    rolling_downside_deviation,
    rolling_excess_kurtosis,
    rolling_hit_rate,
    rolling_max_drawdown,
    rolling_profit_factor,
    rolling_sharpe,
    rolling_skewness,
    rolling_sortino,
    rolling_tracking_error,
    rolling_volatility,
)
from ruin.tail import conditional_value_at_risk, expected_shortfall, value_at_risk
from ruin.volatility import (
    annualize_volatility,
    downside_deviation,
    semi_deviation,
    volatility,
)

__all__ = [
    # returns
    "from_prices",
    "total_return",
    "annualize_return",
    "cagr",
    # volatility
    "volatility",
    "annualize_volatility",
    "downside_deviation",
    "semi_deviation",
    # drawdown
    "drawdown_series",
    "max_drawdown",
    "average_drawdown",
    "max_drawdown_duration",
    "recovery_time",
    "time_underwater",
    "drawdown_start",
    "drawdown_end",
    # ratios
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "information_ratio",
    "treynor_ratio",
    "omega_ratio",
    # tail
    "value_at_risk",
    "conditional_value_at_risk",
    "expected_shortfall",
    # market
    "beta",
    "downside_beta",
    "upside_beta",
    "alpha",
    "tracking_error",
    "correlation",
    "up_capture",
    "down_capture",
    # distribution
    "skewness",
    "excess_kurtosis",
    "jarque_bera",
    "JarqueBeraResult",
    "autocorrelation",
    # activity
    "hit_rate",
    "average_win",
    "average_loss",
    "win_loss_ratio",
    "profit_factor",
    "best_period",
    "worst_period",
    "longest_winning_streak",
    "longest_losing_streak",
    # rolling
    "rolling_volatility",
    "rolling_downside_deviation",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_beta",
    "rolling_correlation",
    "rolling_tracking_error",
    "rolling_alpha",
    "rolling_skewness",
    "rolling_excess_kurtosis",
    "rolling_autocorrelation",
    "rolling_max_drawdown",
    "rolling_hit_rate",
    "rolling_profit_factor",
    # periods
    "mtd",
    "qtd",
    "ytd",
    "trailing",
    "since_inception",
    "periods_per_year_for",
    "annual_to_periodic",
    "periodic_to_annual",
    # inference
    "sharpe_standard_error",
    "sharpe_confidence_interval",
    "bootstrap_metric",
    # report
    "summary",
]
