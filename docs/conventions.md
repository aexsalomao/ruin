# Conventions

Library-wide behaviour: sign conventions, NaN policy, input types, alignment
rules. For per-metric formulas, definitions, assumptions, and citations, see
[`metrics.md`](metrics.md).

## Sign Conventions

### Returns
Returns are dimensionless fractions. A 1% gain is `0.01`, not `1.0`.

### Drawdowns
Drawdowns are **non-positive floats**. A 23% drawdown is `-0.23`. Consistent
with the definition `wealth[t] / hwm[t] - 1`.

### VaR and CVaR
VaR and CVaR are **positive loss magnitudes** (desk convention). A VaR of
`0.02` means "lose at most 2% with the given confidence level."

### Alpha
Alpha is annualized in the same units as the returns (fractions per year). A
positive alpha of `0.03` means 3 percentage points per year of excess return.

## Annualization

All annualization uses `periods_per_year` as an explicit kwarg. It is never
inferred from a time index. Conventional values:

| Frequency | `periods_per_year` |
|-----------|-------------------|
| Daily (trading days) | 252 |
| Weekly | 52 |
| Monthly | 12 |
| Quarterly | 4 |
| Annual | 1 |

Use `ruin.periods.periods_per_year_for("D")` for programmatic access. For the
geometric / arithmetic / sqrt-of-time formulas themselves, see the relevant
entries in [`metrics.md`](metrics.md).

## Risk-Free Rates

All functions that accept `risk_free` expect a **per-period** rate, not an
annualized one. Convert using `ruin.periods.annual_to_periodic`:

```python
rf_annual = 0.05  # 5% annual
rf_daily = ruin.periods.annual_to_periodic(rf_annual, periods_per_year=252)
sharpe = ruin.sharpe_ratio(returns, risk_free=rf_daily, periods_per_year=252)
```

## NaN Handling

By default, all functions **drop NaN and null values** before computation.
This is documented on every function. Use `strict=True` in `summary()` to
raise instead of dropping.

## Input Types

Every function accepts:
- `pl.Series`
- `np.ndarray` (1-D float)
- `pl.DataFrame` (single-column for scalar functions; multi-column for
  `summary()`)

**Not supported:** `pd.Series`, `pd.DataFrame`, `list` (use
`pl.Series(your_list)` first).

## Alignment

When two Series / arrays are passed (e.g., `returns` and `benchmark`):

- Equal length is required; the caller is responsible for alignment.
- NaN removal happens independently before the length check, which means
  NaN-dropping can cause a length mismatch. Pre-clean your data before
  passing to benchmark functions.
