# ruin

A Polars-first risk metrics library for quant hedge funds.

## Installation

```bash
pip install ruin
# or with optional SciPy extras (reserved for future use):
pip install "ruin[stats]"
```

## Quickstart

```python
import polars as pl
import ruin

returns = pl.Series([0.01, -0.02, 0.03, -0.01, 0.02, ...])  # daily returns

sr = ruin.sharpe_ratio(returns, periods_per_year=252)
mdd = ruin.max_drawdown(returns)
df = ruin.summary(returns, periods_per_year=252)
```

## Module Overview

| Module | What it computes |
|--------|-----------------|
| `ruin.returns` | `from_prices`, `total_return`, `annualize_return`, `cagr` |
| `ruin.volatility` | `volatility`, `annualize_volatility`, `downside_deviation`, `semi_deviation` |
| `ruin.drawdown` | `drawdown_series`, `max_drawdown`, `average_drawdown`, `max_drawdown_duration`, `recovery_time`, `time_underwater`, `drawdown_start`, `drawdown_end` |
| `ruin.ratios` | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `information_ratio`, `treynor_ratio`, `omega_ratio` |
| `ruin.tail` | `value_at_risk`, `conditional_value_at_risk` / `expected_shortfall` |
| `ruin.market` | `beta`, `downside_beta`, `upside_beta`, `alpha`, `tracking_error`, `correlation`, `up_capture`, `down_capture` |
| `ruin.distribution` | `skewness`, `excess_kurtosis`, `jarque_bera`, `autocorrelation` |
| `ruin.activity` | `hit_rate`, `average_win`, `average_loss`, `win_loss_ratio`, `profit_factor`, `best_period`, `worst_period`, `longest_winning_streak`, `longest_losing_streak` |
| `ruin.rolling` | Rolling versions of all major metrics, returning length-aligned `pl.Series` |
| `ruin.periods` | `mtd`, `qtd`, `ytd`, `trailing`, `since_inception`, `periods_per_year_for`, `annual_to_periodic`, `periodic_to_annual` |
| `ruin.inference` | `sharpe_standard_error`, `sharpe_confidence_interval`, `bootstrap_metric` |
| `ruin.report` | `summary` — the one bundled function |

## Composition Principle

**This library provides building blocks. If you want a bundled metric, call `summary()` or compose the primitives yourself. Every function returns exactly one thing. We will not add bundled convenience functions.**

Composition in Polars is cheap. A custom bundle is two lines:

```python
import polars as pl
import ruin

# Rolling Sharpe with bootstrap confidence intervals
returns = pl.Series(your_returns)

# Rolling Sharpe (returns a Series)
rs = ruin.rolling_sharpe(returns, window=60, periods_per_year=252)

# Bootstrap CI on the scalar Sharpe
point, lo, hi = ruin.bootstrap_metric(
    lambda r: ruin.sharpe_ratio(r, periods_per_year=252),
    returns,
    n_samples=1000,
    confidence=0.95,
)
print(f"Sharpe: {point:.3f} ({lo:.3f}, {hi:.3f})")
```

Call multiple functions and compose:

```python
metrics = {
    "sharpe": ruin.sharpe_ratio(returns, periods_per_year=252),
    "sortino": ruin.sortino_ratio(returns, periods_per_year=252),
    "max_dd": ruin.max_drawdown(returns),
    "cvar": ruin.conditional_value_at_risk(returns),
}
```

## What This Library Is Not

The following are **explicitly out of scope**. Do not open issues requesting them.

- **Money-weighted returns / IRR / fee crystallization** — needs cash flows; belongs in a sibling library.
- **Portfolio construction** — optimization, risk parity, mean-variance, Black-Litterman.
- **Factor models** — Fama-French, Barra-style attribution, Brinson attribution.
- **Backtesting engine** — trade simulation, transaction costs, execution modeling.
- **Data fetching** — no yfinance, no Bloomberg, no CSV readers.
- **Time series hygiene** — resampling, calendar alignment, gap filling, corporate actions.
- **Stress testing / Monte Carlo / scenario analysis**.
- **Forecasting** — no GARCH, EWMA, regime models, volatility prediction.
- **Plotting** — use Polars + Altair/Plotly/Matplotlib directly.
- **Pandas support** — use Polars. `pd.Series.to_numpy()` converts in one call if needed.
- **Cornish-Fisher VaR** — reserved for a future `ruin[stats]` extra.

## Design Principles

1. **Returns in, numbers out.** No prices, no positions, no trades, no I/O.
2. **Polars-first, minimal deps.** Runtime: `polars` + `numpy` only.
3. **Pure functions.** No side effects, no hidden state.
4. **One function, one metric, one return type.** Scalar → `float`, rolling → `pl.Series`.
5. **Explicit over implicit.** `periods_per_year` is always explicit. `risk_free` is per-period.
6. **NaN handling.** Drop by default; `strict=True` in `summary()` raises instead.

## Sign Conventions

- Drawdowns are non-positive: `-0.23` means 23% drawdown.
- VaR and CVaR are positive loss magnitudes (desk convention): `0.02` means "lose at most 2%."

See [`docs/conventions.md`](docs/conventions.md) for full details.
See [`docs/assumptions.md`](docs/assumptions.md) for when each metric can mislead you.
