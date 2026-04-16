# Conventions

## Sign Conventions

### Returns
Returns are dimensionless fractions. A 1% gain is `0.01`, not `1.0`.

### Drawdowns
Drawdowns are **non-positive floats**. A 23% drawdown is `-0.23`.
This is consistent with the definition: `wealth[t] / hwm[t] - 1`.

### VaR and CVaR
VaR and CVaR are **positive loss magnitudes** (desk convention).
A VaR of `0.02` means "lose at most 2% with the given confidence level."
This is the convention used by most risk desks: losses are positive numbers.

### Alpha
Alpha is annualized in the same units as the returns (fractions per year).
A positive alpha of `0.03` means 3 percentage points per year of excess return.

## Annualization

All annualization uses `periods_per_year` as an explicit kwarg. It is never inferred
from a time index. Common values:

| Frequency | `periods_per_year` |
|-----------|-------------------|
| Daily (trading days) | 252 |
| Weekly | 52 |
| Monthly | 12 |
| Quarterly | 4 |
| Annual | 1 |

Use `ruin.periods.periods_per_year_for("D")` etc. for programmatic access.

### Volatility annualization
`annualized_vol = periodic_vol * sqrt(periods_per_year)`

This is the sqrt-of-time rule. It assumes i.i.d. returns. If returns have autocorrelation,
the correct formula is:

```
effective_ppy = periods_per_year * (1 + 2 * sum_{k=1}^{K} rho_k)
annualized_vol_corrected = periodic_vol * sqrt(effective_ppy)
```

`ruin` does not implement autocorrelation-corrected volatility annualization in v1,
but provides the building blocks (`autocorrelation`) to implement it yourself.

### Return annualization
Geometric (default): `(1 + total_return)^(periods_per_year / n) - 1`
Arithmetic: `mean(r) * periods_per_year`

Use geometric unless you have a specific reason not to (arithmetic overstates expected
terminal wealth for volatile series).

## Risk-Free Rates

All functions that accept `risk_free` expect a **per-period** rate, not an annualized one.
Convert using `ruin.periods.annual_to_periodic`:

```python
rf_annual = 0.05  # 5% annual
rf_daily = ruin.periods.annual_to_periodic(rf_annual, periods_per_year=252)
sharpe = ruin.sharpe_ratio(returns, risk_free=rf_daily, periods_per_year=252)
```

## NaN Handling

By default, all functions **drop NaN and null values** before computation. This is
documented on every function.

Use `strict=True` in `summary()` to raise instead of dropping.

## Input Types

Every function accepts:
- `pl.Series`
- `np.ndarray` (1-D float)
- `pl.DataFrame` (single-column for scalar functions; multi-column for `summary()`)

**Not supported:** `pd.Series`, `pd.DataFrame`, `list` (use `pl.Series(your_list)` first).

## Alignment

When two Series/arrays are passed (e.g., `returns` and `benchmark`):
- Equal length is required; the caller is responsible for alignment.
- NaN removal happens independently before the length check, which means NaN-dropping
  can cause a length mismatch. Pre-clean your data before passing to benchmark functions.
