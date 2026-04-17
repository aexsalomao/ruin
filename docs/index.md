---
hide:
  - navigation
  - toc
---

<div align="center" markdown>

# ruin

**Polars-first risk metrics for quant hedge funds — returns in, numbers out.**

[![PyPI](https://img.shields.io/pypi/v/ruin?color=red)](https://pypi.org/project/ruin/)
[![Python](https://img.shields.io/pypi/pyversions/ruin)](https://pypi.org/project/ruin/)
[![License](https://img.shields.io/github/license/aexsalomao/ruin)](https://github.com/aexsalomao/ruin/blob/master/LICENSE)
[![CI](https://github.com/aexsalomao/ruin/actions/workflows/ci.yml/badge.svg)](https://github.com/aexsalomao/ruin/actions/workflows/ci.yml)
[![Docs](https://github.com/aexsalomao/ruin/actions/workflows/docs.yml/badge.svg)](https://aexsalomao.github.io/ruin/)

```bash
pip install ruin
```

</div>

---

## Features

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Every metric you need**

    ---

    Sharpe, Sortino, Calmar, Information, Treynor, Omega, max-drawdown, VaR, CVaR, beta (up/down), alpha, tracking error, capture, skew, kurtosis, Jarque-Bera, hit rate, profit factor, streaks — and rolling versions of the major ones.

    [:octicons-arrow-right-24: API Reference](api.md)

-   :material-lightning-bolt:{ .lg .middle } **Polars-first, minimal deps**

    ---

    Runtime: `polars` + `numpy` only. Public Polars outputs are `Float32`; internal math stays in `Float64` to avoid catastrophic cancellation.

-   :material-puzzle-outline:{ .lg .middle } **Composition over convenience**

    ---

    Every function returns exactly one thing. `summary()` is the only bundled call. Compose primitives in two lines instead of pulling 30 wrappers off the shelf.

-   :material-function-variant:{ .lg .middle } **Pure functions**

    ---

    No hidden state, no I/O, no plotting, no data fetching. Returns in, numbers out — testable by construction.

-   :material-shield-check:{ .lg .middle } **Tested rigorously**

    ---

    Hand-computed reference values, Hypothesis property tests, and `pytest-benchmark` regression coverage. Every metric is documented with formula and source.

-   :material-format-list-numbered:{ .lg .middle } **Explicit conventions**

    ---

    `periods_per_year` is always explicit. `risk_free` is per-period. Drawdowns are non-positive. VaR / CVaR are positive loss magnitudes. NaN handling documented per function.

</div>

---

## Quick start

=== "Scalars"

    ```python
    import polars as pl
    import ruin

    returns = pl.Series([0.01, -0.02, 0.03, -0.01, 0.02])

    ruin.sharpe_ratio(returns, periods_per_year=252)
    ruin.sortino_ratio(returns, periods_per_year=252)
    ruin.max_drawdown(returns)
    ruin.value_at_risk(returns, confidence=0.95)
    ruin.conditional_value_at_risk(returns, confidence=0.95)
    ```

=== "Rolling"

    ```python
    import polars as pl
    import ruin

    returns = pl.Series(your_daily_returns)

    rs = ruin.rolling_sharpe(returns, window=60, periods_per_year=252)
    rv = ruin.rolling_volatility(returns, window=60, periods_per_year=252)
    rdd = ruin.rolling_max_drawdown(returns, window=252)
    ```

=== "Summary"

    ```python
    import polars as pl
    import ruin

    returns = pl.Series(your_daily_returns)

    df = ruin.summary(returns, periods_per_year=252)
    print(df)
    ```

=== "Bootstrap CI"

    ```python
    import polars as pl
    import ruin

    returns = pl.Series(your_daily_returns)

    point, lo, hi = ruin.bootstrap_metric(
        lambda r: ruin.sharpe_ratio(r, periods_per_year=252),
        returns,
        n_samples=1000,
        confidence=0.95,
    )
    print(f"Sharpe: {point:.3f} ({lo:.3f}, {hi:.3f})")
    ```

<div align="center" markdown>

[:octicons-arrow-right-24: Full API Reference](api.md){ .md-button .md-button--primary }
[:octicons-arrow-right-24: Conventions](conventions.md){ .md-button }
[:octicons-arrow-right-24: Metrics & formulas](metrics.md){ .md-button }

</div>

---

## Design principles

1. **Returns in, numbers out.** No prices, no positions, no trades, no I/O.
2. **Polars-first, minimal deps.** Runtime: `polars` + `numpy` only.
3. **Pure functions.** No side effects, no hidden state.
4. **One function, one metric, one return type.** Scalar → `float`, rolling → `pl.Series`.
5. **Explicit over implicit.** `periods_per_year` is always explicit. `risk_free` is per-period.
6. **NaN handling.** Drop by default; `strict=True` in `summary()` raises instead.

## What ruin is not

The following are **explicitly out of scope** — please do not open issues requesting them:

- Money-weighted returns / IRR / fee crystallization
- Portfolio construction (optimization, risk parity, mean-variance, Black-Litterman)
- Factor models (Fama-French, Barra, Brinson attribution)
- Backtesting engine (trade simulation, transaction costs, execution modeling)
- Data fetching (no yfinance, no Bloomberg, no CSV readers)
- Time series hygiene (resampling, calendar alignment, gap filling)
- Stress testing / Monte Carlo / scenario analysis
- Forecasting (GARCH, EWMA, regime models)
- Plotting
- Pandas support — use Polars; `pd.Series.to_numpy()` converts in one call if needed
