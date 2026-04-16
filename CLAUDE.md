# CLAUDE.md — ruin

Guidance for Claude Code when working in this repository. Keep this file short and prescriptive — global defaults live in `C:\Users\axsal\dev\CLAUDE.md`; only repo-specific rules belong here.

## What this repo is

`ruin` is a Polars-first risk-metrics library for quant hedge funds. Scope is strict: **returns in, numbers out**. No prices, positions, trades, I/O, backtesting, optimization, or plotting. See `README.md` for the complete exclusion list — do not grow scope.

The composition principle is load-bearing: every function returns exactly one thing. The only bundled function is `ruin.report.summary()`. **Do not add convenience wrappers.**

## Layout

```
src/ruin/
  _internal/validate.py    # dtype constants, input coercion, NaN policy, length checks
  returns.py               # total_return, annualize_return, cagr, from_prices
  volatility.py            # volatility, annualize_volatility, downside/semi-deviation
  drawdown.py              # drawdown_series, max_drawdown, durations, recovery
  ratios.py                # sharpe, sortino, calmar, treynor, omega, information
  tail.py                  # value_at_risk, conditional_value_at_risk
  market.py                # beta (up/down), alpha, tracking_error, capture, correlation
  distribution.py          # skewness, excess_kurtosis, jarque_bera, autocorrelation
  activity.py              # hit_rate, profit_factor, streaks, best/worst period
  rolling.py               # rolling_* variants — all outputs Float32
  periods.py               # mtd/qtd/ytd/trailing, rate conversion, periods_per_year_for
  inference.py             # Lo (2002) Sharpe SE/CI, bootstrap_metric
  report.py                # summary() — the one bundled function
tests/
  conftest.py              # fixtures + FLOAT32_{REL,ABS}_TOL constants
  test_*.py                # one module per source module
  test_properties.py       # hypothesis-based invariants
  test_reference.py        # hand-computed locked values
  test_benchmarks.py       # pytest-benchmark regression tests
```

## Non-negotiable conventions

### Dtype policy (Float32 out, Float64 in)
- **Public Polars outputs are Float32.** `Series`/`DataFrame` results from `drawdown_series`, `rolling_*`, `from_prices`, `summary`, etc. must be cast to `FLOAT_DTYPE` before return.
- **Internal math stays in Float64** to avoid catastrophic cancellation in `cum_prod`, variance accumulators, and autocorrelation adjustments. `to_series` / `to_dataframe` cast inputs to `INTERNAL_FLOAT_DTYPE` on entry.
- Both constants live in `src/ruin/_internal/validate.py`:
  ```python
  FLOAT_DTYPE: pl.DataType = pl.Float32
  INTERNAL_FLOAT_DTYPE: pl.DataType = pl.Float64
  ```
- Scalar Python `float` returns stay as-is (Float64 semantics). Only cast when emitting a Polars container.
- In tests, use `FLOAT32_REL_TOL = 1e-5` / `FLOAT32_ABS_TOL = 1e-6` from `tests/conftest.py` whenever comparing Float32 outputs. Float64 scalars can use `1e-9` — `1e-9` on a Float32 value is noise.

### Typing
- Python 3.10+ native generics only: `dict`, `list`, `tuple`, `set`, `type`. **No `typing.Dict`/`List`/`Tuple`/`Optional`/`Union`.**
- Use pipe unions (`X | Y`, `X | None`). The `ReturnInput` alias is `pl.Series | pl.Expr | np.ndarray | pl.DataFrame`.
- Import `Callable` from `collections.abc`, not `typing`.
- All public functions are fully annotated. Frozen dataclasses (`JarqueBeraResult`) for structured returns.
- Every module starts with `from __future__ import annotations`.

### Input handling
- All public functions accept `ReturnInput` and run through `to_series` (or `to_dataframe` for multi-stream paths). Never branch on input type inside metric code.
- NaN/null are dropped in `to_series`. Strict mode (`summary(strict=True)`) checks and raises before coercion.
- Length/positivity validation uses helpers in `_internal/validate.py` (`require_minimum_length`, `require_same_length`, `require_strictly_positive`). Don't re-implement.

### Numerical guards
- Guard degenerate inputs with `r.n_unique() <= 1` in addition to `sigma == 0.0`. Polars `std` on a constant series can return a tiny nonzero value; `n_unique` is the ground truth. Applied in `skewness` and `excess_kurtosis`; extend to any new moment-based metric.
- `annualize_return` returns NaN when total return ≤ -1 (ruin case) rather than raising — downstream plotting/reporting shouldn't blow up on a wiped-out track record.
- Narrow `try/except` in `report._safe` to `(ValueError, ZeroDivisionError)`. Don't swallow broad `Exception`.

### Rolling implementation
- `rolling.py` exposes helpers: `_require_int_window`, `_require_matching_lengths`, `_window_apply`. Path-dependent metrics (skew, kurtosis, max-drawdown, profit-factor) go through `_window_apply`; simple reductions use Polars' native `rolling_std`/`rolling_mean`/`rolling_cov`.
- Every `rolling_*` return ends with `.cast(FLOAT_DTYPE).rename("rolling_<name>")`.

## Testing

- `pytest --no-cov -q` for a fast run. `pytest` (default) includes coverage + benchmarks.
- 258 tests today; keep additions cohesive with the existing classes in each `test_*.py`.
- Three test tiers:
  1. Unit tests per module with edge cases, invalid-input branches, and dtype assertions.
  2. `test_reference.py` locks exact hand-computed values. Changes here require deliberate review.
  3. `test_properties.py` — hypothesis invariants. If a property fails under hypothesis, **recompute the invariant** before suppressing; a real bug is far likelier than a hypothesis flake.
- `test_benchmarks.py` is regression-only — don't assert absolute wall-time.
- When hardening: prefer asserting dtype (`result.dtype == pl.Float32`) + tolerance (`math.isclose(..., rel_tol=FLOAT32_REL_TOL)`) over bare equality.

## Dependencies

- Runtime: `polars` and `numpy` only. Adding any runtime dep requires user sign-off.
- `ruin[stats]` is reserved for future SciPy-backed additions (Cornish-Fisher VaR, etc.) — not yet wired.
- Dev: `pytest`, `pytest-cov`, `pytest-benchmark`, `hypothesis`, `ruff`, `mypy`.

## Style

- Defer to `C:\Users\axsal\dev\CLAUDE.md` for global defaults (Polars preference, memory efficiency, testability).
- Comments: only when the *why* is non-obvious — a numerical guard, a sign convention, a Polars quirk. Identifier names carry the *what*.
- Docstrings on every public function with Parameters / Returns / Notes. Keep them tight; point at `docs/conventions.md` / `docs/assumptions.md` for long-form discussion.
- Sign conventions (locked):
  - Drawdowns non-positive (`-0.23` = 23% drawdown).
  - VaR / CVaR positive loss magnitudes (`0.02` = "lose at most 2%").
