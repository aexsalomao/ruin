# Assumptions and When Each Metric Lies

## Returns

**`total_return`** — Assumes compounding. Returns a path-dependent number; does not reflect risk.

**`annualize_return`** (geometric) — Assumes the return series is representative of a full year and that compounding continues at the same rate. Lies badly on short series (< 1 year) where realized variance is high.

**`annualize_return`** (arithmetic) — Assumes i.i.d. returns with no compounding effects. Overstates expected terminal wealth for volatile strategies.

**`cagr`** — Same as geometric `annualize_return`. Overstates skill when the series is lucky (high ending value) or short.

## Volatility

**`volatility` / `annualize_volatility`** — Assumes i.i.d. normally distributed returns. The sqrt-of-time annualization formula (`sigma * sqrt(T)`) is **invalid** when returns have autocorrelation. Use `ruin.distribution.autocorrelation` to check before trusting annualized volatility.

**`downside_deviation`** — Counts all periods in the denominator (including up-periods). This is the Sortino/Upside Potential convention. An alternative is to use only downside observations in the denominator (use `semi_deviation` for that).

## Drawdown

**`max_drawdown`** — Path-dependent. Short series may not capture the true worst drawdown. Also depends heavily on which periods are included; month-end vs. daily data will produce different numbers.

**`average_drawdown`** — Sensitive to how you define an "episode." This library uses contiguous underwater runs. Other implementations aggregate differently.

**`recovery_time`** — Returns `NaN` if the portfolio has not recovered. This is intentional: pretending a strategy recovered when it hasn't is dishonest.

## Ratios

**`sharpe_ratio`** — Assumes normally distributed returns (used to justify annualization). Positive Sharpe can be manufactured by skewing returns or by using stale/smoothed pricing. Use `autocorrelation`, `skewness`, and `excess_kurtosis` as diagnostics.

**`sortino_ratio`** — Sensitive to the choice of threshold (MAR). Different desks use different conventions.

**`calmar_ratio`** — Uses max drawdown, which is path-dependent and sensitive to lookback period. Short histories with no large drawdowns produce artificially high Calmar.

**`treynor_ratio`** — Divides by beta. Poorly defined when beta is near zero or negative. Only meaningful for long-only strategies in a CAPM framework.

**`omega_ratio`** at threshold=0 is equivalent to profit_factor and to "gain-to-pain ratio." Using a threshold equal to the risk-free rate is more economically meaningful.

## Tail Risk

**`value_at_risk` (historical)** — Nonparametric. Requires large samples to be stable. Insensitive to the exact shape of the tail beyond the quantile cutoff.

**`value_at_risk` (parametric)** — Assumes normality. Systematically underestimates loss in fat-tailed return distributions. Use historical VaR for checking; use parametric as a sanity check.

**`conditional_value_at_risk`** — More stable than VaR and coherent (sub-additive). Still sensitive to sample size; bootstrap it using `ruin.inference.bootstrap_metric` to get confidence intervals.

## Market Metrics

**`beta`** — OLS slope. Assumes linear relationship between strategy and benchmark. Nonstationarity makes rolling beta more informative than full-sample beta for live monitoring.

**`alpha`** — Jensen's alpha: the CAPM intercept. Valid only if CAPM holds, which it generally doesn't. Treat as a performance decomposition, not a causal attribution.

**`up_capture` / `down_capture`** — Geometric compounding over up/down benchmark periods. Results depend heavily on how many up/down periods are in the sample; short series are unreliable.

## Distribution

**`skewness` / `excess_kurtosis`** — Require large samples to be reliable (n > 100 for kurtosis). Biased-correction formulas assume independent observations.

**`jarque_bera`** — Asymptotic test; unreliable for small samples (< ~100 observations). A high p-value does not prove normality; it merely fails to reject it.

**`autocorrelation`** — Lag-1 autocorrelation is a key diagnostic for smoothed returns (illiquid strategies, hedge funds with manual pricing). Positive autocorrelation invalidates the sqrt-of-time rule and inflates Sharpe.

## Inference

**`sharpe_standard_error`** — Lo (2002) formula adjusts for first-order autocorrelation only. If returns have higher-order serial correlation, this SE is still biased.

**`bootstrap_metric`** — Resamples with replacement, which destroys serial dependence. Appropriate for i.i.d. returns; use a block bootstrap (not provided) for autocorrelated series.
