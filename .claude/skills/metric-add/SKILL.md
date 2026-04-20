---
name: metric-add
description: Add a new risk metric to ruin — pick the right module, write a pure Polars-first implementation, add rolling/summary entries, and cover it with invariants and reference-value tests. Use when the user asks to add a metric, ratio, drawdown variant, distribution stat, or performance measure.
---

# Add a Risk Metric

ruin is a Polars-first risk library. A new metric must be pure, stateless, and Polars-native. No pandas.

## Pick the module

One of:
- `returns.py` — return transforms (`from_prices`, `total_return`, `annualize_return`, `cagr`)
- `volatility.py` — dispersion (`volatility`, `annualize_volatility`, `downside_deviation`, `semi_deviation`)
- `drawdown.py` — drawdowns and underwater time
- `ratios.py` — risk-adjusted ratios (Sharpe, Sortino, Calmar, Information, Treynor, Omega)
- `tail.py` — VaR / CVaR / expected shortfall
- `market.py` — beta, alpha, tracking error, capture ratios, correlation
- `distribution.py` — skewness, kurtosis, Jarque-Bera, autocorrelation
- `activity.py` — hit rate, win/loss, streaks
- `rolling.py` — rolling variant (see below — always needed)
- `report.py` — summary-table entry (see below — always needed)

If the metric doesn't fit one module, it's really two things.

## API contract

```python
def metric_name(
    returns: pl.Series,                 # or prices: pl.Series for price-input metrics
    *,
    periods_per_year: int = 252,        # only if the metric is annualized
    # other metric-specific kwargs with sensible defaults
) -> float:                              # or pl.Series for series-valued metrics
```

Rules:
- **Positional arg is the data.** Everything else is keyword-only (`*`).
- **Input is `pl.Series` or `pl.DataFrame`.** Never `np.ndarray`, `list`, or `pd.Series` — let the caller convert.
- **Output is a scalar `float`** for single-number metrics, `pl.Series` for path-valued metrics (drawdown series, rolling).
- **Lazy when it helps.** If the computation chains 3+ Polars ops, use `.lazy()` and `.collect()` at the end.
- **No side effects.** No logging inside the metric (caller can log the result).

## Always update these alongside the metric

1. **`rolling.py`** — add a rolling variant if the metric makes sense on a window:
   ```python
   def rolling_metric_name(returns: pl.Series, window: int, **kwargs) -> pl.Series:
       ...
   ```
   Length-aligned output (pad with nulls at the start, no silent truncation).

2. **`report.py::summary`** — add a row in the summary table so `ruin.summary(returns)` includes the new metric.

3. **`__init__.py`** — re-export from the module so `ruin.metric_name(...)` works without the submodule path.

4. **README module-overview table** — add the method to its module's row.

## Tests

Put tests in `tests/test_<module>.py`. You need at least:

### Reference-value tests
Three hand-verified cases with a deterministic input series:
```python
@pytest.mark.parametrize("series,expected", [
    (pl.Series([0.01, -0.02, 0.03]), 0.006123),   # from a spreadsheet or known-good lib
    ...
], ids=["mixed", "all-positive", "all-negative"])
def test_metric_name_returns_known_value(series, expected):
    assert ruin.metric_name(series) == pytest.approx(expected, rel=1e-6)
```

### Invariant tests (property-based when you can)
Use `hypothesis` for structural properties — see `~/.claude/skills/property-based-testing/` for patterns. Examples:
- Drawdowns are always `≤ 0`.
- `annualize_return(r, n)` then de-annualize returns identity.
- Scaling returns by `c` scales volatility by `|c|`.
- Sharpe of a constant series is well-defined or `NaN` (pick one, document it).

### Edge cases
- Empty series → raise or return `NaN`, not silently 0.
- Single observation → what's the contract? Document and test.
- All-NaN input → raise with a clear message.
- Mismatched series lengths (for two-input metrics like beta) → raise.

### Numerical tests
- Use `pytest.approx` for floats — never `==`.
- Test against a reference implementation (`empyrical`, `quantstats`, or spreadsheet values) where possible. See `numerical-parity` skill.

## Checklist

- [ ] Pure function, Polars-native, typed signature.
- [ ] Docstring with `Args`/`Returns`/`Raises` when behavior is non-trivial.
- [ ] Rolling variant in `rolling.py` (if meaningful).
- [ ] Row added in `report.py::summary`.
- [ ] Re-exported in `__init__.py`.
- [ ] Reference-value, invariant, and edge-case tests.
- [ ] `CHANGELOG.md` → `## [Unreleased]` → `### Added`.
- [ ] `uv run ruff check && uv run ruff format --check && uv run mypy --strict src && uv run pytest` all green.

## What not to do

- Don't convert to `np.ndarray` in the middle of a computation. Stay in Polars.
- Don't add `Any` to escape a type error — fix the type.
- Don't add the metric without a rolling variant if the metric is meaningful on a window. The API would be inconsistent.
- Don't compare against `pandas`/`numpy` outputs without understanding the convention (biased vs. unbiased variance, log vs. simple returns, periods_per_year default). Pick a convention and document it.
