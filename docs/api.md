# API Reference

All functions accept `pl.Series`, `np.ndarray`, or single-column `pl.DataFrame` as the return input. NaN / null are dropped before computation. See [Conventions](conventions.md) for sign conventions, NaN policy, and alignment rules. See [Metrics](metrics.md) for formulas, assumptions, and citations.

---

## Returns (`ruin.returns`)

| Call | Returns |
|---|---|
| `ruin.from_prices(prices, *, log=False)` | `pl.Series` — `Float32` period returns of length $N - 1$ |
| `ruin.total_return(returns)` | `float` — compounded total return |
| `ruin.annualize_return(returns, *, periods_per_year)` | `float` — geometric annualized return; `NaN` when total return ≤ -1 |
| `ruin.cagr(prices, *, periods_per_year)` | `float` — compound annual growth rate from a price series |

---

## Volatility (`ruin.volatility`)

| Call | Returns |
|---|---|
| `ruin.volatility(returns)` | `float` — sample standard deviation |
| `ruin.annualize_volatility(vol, *, periods_per_year)` | `float` — `vol * sqrt(periods_per_year)` |
| `ruin.downside_deviation(returns, *, threshold=0.0)` | `float` — std of returns below `threshold` |
| `ruin.semi_deviation(returns)` | `float` — `downside_deviation` with `threshold=mean(returns)` |

---

## Drawdown (`ruin.drawdown`)

| Call | Returns |
|---|---|
| `ruin.drawdown_series(returns)` | `pl.Series` — `Float32`, non-positive (`-0.23` = 23%) |
| `ruin.max_drawdown(returns)` | `float` — most negative drawdown value |
| `ruin.average_drawdown(returns)` | `float` — mean of distinct drawdown episodes |
| `ruin.max_drawdown_duration(returns)` | `int` — length (in periods) of the longest drawdown |
| `ruin.recovery_time(returns)` | `int` — periods from trough to recovery (0 if not yet recovered) |
| `ruin.time_underwater(returns)` | `int` — total periods spent below the high-water mark |
| `ruin.drawdown_start(returns)` | `int` — index at which the max-drawdown episode began |
| `ruin.drawdown_end(returns)` | `int` — index of the trough |

---

## Ratios (`ruin.ratios`)

| Call | Returns |
|---|---|
| `ruin.sharpe_ratio(returns, *, risk_free=0.0, periods_per_year)` | `float` — annualized |
| `ruin.sortino_ratio(returns, *, threshold=0.0, periods_per_year)` | `float` — annualized |
| `ruin.calmar_ratio(returns, *, periods_per_year)` | `float` — annualized return / |max drawdown| |
| `ruin.information_ratio(returns, benchmark, *, periods_per_year)` | `float` — annualized |
| `ruin.treynor_ratio(returns, benchmark, *, risk_free=0.0, periods_per_year)` | `float` — annualized |
| `ruin.omega_ratio(returns, *, threshold=0.0)` | `float` — gain / loss probability-weighted ratio |

---

## Tail (`ruin.tail`)

VaR and CVaR are **positive loss magnitudes** (desk convention).

| Call | Returns |
|---|---|
| `ruin.value_at_risk(returns, *, confidence=0.95)` | `float` — historical VaR |
| `ruin.conditional_value_at_risk(returns, *, confidence=0.95)` | `float` — historical expected shortfall |
| `ruin.expected_shortfall(returns, *, confidence=0.95)` | `float` — alias for `conditional_value_at_risk` |

---

## Market (`ruin.market`)

Benchmark inputs must be pre-aligned in length with `returns`.

| Call | Returns |
|---|---|
| `ruin.beta(returns, benchmark)` | `float` |
| `ruin.downside_beta(returns, benchmark, *, threshold=0.0)` | `float` |
| `ruin.upside_beta(returns, benchmark, *, threshold=0.0)` | `float` |
| `ruin.alpha(returns, benchmark, *, risk_free=0.0, periods_per_year)` | `float` — annualized |
| `ruin.tracking_error(returns, benchmark, *, periods_per_year)` | `float` — annualized |
| `ruin.correlation(returns, benchmark)` | `float` — Pearson |
| `ruin.up_capture(returns, benchmark)` | `float` |
| `ruin.down_capture(returns, benchmark)` | `float` |

---

## Distribution (`ruin.distribution`)

| Call | Returns |
|---|---|
| `ruin.skewness(returns)` | `float` — sample skewness |
| `ruin.excess_kurtosis(returns)` | `float` — sample kurtosis − 3 |
| `ruin.jarque_bera(returns)` | `JarqueBeraResult` — frozen dataclass with `statistic`, `p_value` |
| `ruin.autocorrelation(returns, *, lag=1)` | `float` — sample autocorrelation at the given lag |

---

## Activity (`ruin.activity`)

| Call | Returns |
|---|---|
| `ruin.hit_rate(returns)` | `float` — fraction of strictly positive periods |
| `ruin.average_win(returns)` | `float` — mean of positive returns |
| `ruin.average_loss(returns)` | `float` — mean of negative returns |
| `ruin.win_loss_ratio(returns)` | `float` — `average_win / |average_loss|` |
| `ruin.profit_factor(returns)` | `float` — sum of gains / |sum of losses| |
| `ruin.best_period(returns)` | `float` — max return |
| `ruin.worst_period(returns)` | `float` — min return |
| `ruin.longest_winning_streak(returns)` | `int` |
| `ruin.longest_losing_streak(returns)` | `int` |

---

## Rolling (`ruin.rolling`)

All rolling outputs are length-aligned `pl.Series` cast to `Float32` and renamed `rolling_<name>`.

| Call | Returns |
|---|---|
| `ruin.rolling_volatility(returns, *, window, periods_per_year)` | `pl.Series` |
| `ruin.rolling_downside_deviation(returns, *, window, threshold=0.0)` | `pl.Series` |
| `ruin.rolling_sharpe(returns, *, window, risk_free=0.0, periods_per_year)` | `pl.Series` |
| `ruin.rolling_sortino(returns, *, window, threshold=0.0, periods_per_year)` | `pl.Series` |
| `ruin.rolling_beta(returns, benchmark, *, window)` | `pl.Series` |
| `ruin.rolling_correlation(returns, benchmark, *, window)` | `pl.Series` |
| `ruin.rolling_tracking_error(returns, benchmark, *, window, periods_per_year)` | `pl.Series` |
| `ruin.rolling_alpha(returns, benchmark, *, window, risk_free=0.0, periods_per_year)` | `pl.Series` |
| `ruin.rolling_skewness(returns, *, window)` | `pl.Series` |
| `ruin.rolling_excess_kurtosis(returns, *, window)` | `pl.Series` |
| `ruin.rolling_autocorrelation(returns, *, window, lag=1)` | `pl.Series` |
| `ruin.rolling_max_drawdown(returns, *, window)` | `pl.Series` |
| `ruin.rolling_hit_rate(returns, *, window)` | `pl.Series` |
| `ruin.rolling_profit_factor(returns, *, window)` | `pl.Series` |

---

## Periods (`ruin.periods`)

| Call | Returns |
|---|---|
| `ruin.mtd(df, *, date_col)` | `pl.DataFrame` — month-to-date slice |
| `ruin.qtd(df, *, date_col)` | `pl.DataFrame` — quarter-to-date slice |
| `ruin.ytd(df, *, date_col)` | `pl.DataFrame` — year-to-date slice |
| `ruin.trailing(df, *, date_col, days)` | `pl.DataFrame` — trailing-N-days slice |
| `ruin.since_inception(df, *, date_col)` | `pl.DataFrame` — full history |
| `ruin.periods_per_year_for(freq)` | `int` — `"D"`/`"W"`/`"M"`/`"Q"`/`"A"` → `252`/`52`/`12`/`4`/`1` |
| `ruin.annual_to_periodic(rate, *, periods_per_year)` | `float` |
| `ruin.periodic_to_annual(rate, *, periods_per_year)` | `float` |

---

## Inference (`ruin.inference`)

| Call | Returns |
|---|---|
| `ruin.sharpe_standard_error(returns, *, periods_per_year)` | `float` — Lo (2002) IID SE |
| `ruin.sharpe_confidence_interval(returns, *, periods_per_year, confidence=0.95)` | `tuple[float, float]` |
| `ruin.bootstrap_metric(fn, returns, *, n_samples=1000, confidence=0.95, seed=None)` | `tuple[float, float, float]` — `(point, lo, hi)` |

---

## Report (`ruin.report`)

The **only** bundled function in the library.

| Call | Returns |
|---|---|
| `ruin.summary(returns, *, periods_per_year, strict=False)` | `pl.DataFrame` — `Float32` summary across the major metrics |

`strict=True` raises on NaN inputs instead of dropping them.
